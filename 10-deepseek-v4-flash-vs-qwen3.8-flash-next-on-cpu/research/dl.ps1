param([string]$Quant = "UD-IQ1_S", [int]$Shards = 3)
$root  = "D:\Claude AI\AI Model Performance\deepseek-v4-vs-flash-next-1bit"
$dest  = Join-Path $root "models\$Quant"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$base  = "https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/$Quant"
$jobs = @()
for ($i = 1; $i -le $Shards; $i++) {
    $n    = "{0:d5}-of-{1:d5}" -f $i, $Shards
    $file = "DeepSeek-V4-Flash-0731-$Quant-$n.gguf"
    $url  = "$base/$file"
    $out  = Join-Path $dest $file
    $jobs += Start-Job -ScriptBlock {
        param($u, $o)
        curl.exe -sL --retry 10 --retry-delay 5 --retry-all-errors -C - -o $o $u
        $LASTEXITCODE
    } -ArgumentList $url, $out
}
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$jobs | Wait-Job | Out-Null
$sw.Stop()
$jobs | ForEach-Object { "job $($_.Id) exit=$(Receive-Job $_)" }
$tot = (Get-ChildItem $dest -Filter *.gguf | Measure-Object Length -Sum).Sum
"TOTAL_GB=$([math]::Round($tot/1GB,2)) SECONDS=$([math]::Round($sw.Elapsed.TotalSeconds,0)) AVG_MBps=$([math]::Round($tot/1MB/$sw.Elapsed.TotalSeconds,1))"
Get-ChildItem $dest -Filter *.gguf | ForEach-Object { "  $($_.Name)  $([math]::Round($_.Length/1GB,3)) GB" }

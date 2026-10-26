"""Report which processes currently hold GPU memory, biggest first.

Used to check whether a background Ollama (or anything else) is holding VRAM
while a llama.cpp config is being measured.
"""

import subprocess

PS = r"""
$s = (Get-Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction SilentlyContinue).CounterSamples
$s | Where-Object { $_.CookedValue -gt 20MB } | ForEach-Object {
  if ($_.InstanceName -match 'pid_(\d+)_') {
    $pid_ = [int]$Matches[1]
    $p = Get-Process -Id $pid_ -ErrorAction SilentlyContinue
    '{0,9:N0} MiB  {1,-28} pid {2}' -f ($_.CookedValue/1MB), $(if($p){$p.ProcessName}else{'?'}), $pid_
  }
} | Sort-Object -Descending
"""

out = subprocess.run(
    ["powershell", "-NoProfile", "-NonInteractive", "-Command", PS],
    capture_output=True,
    text=True,
    timeout=60,
).stdout
print(out)

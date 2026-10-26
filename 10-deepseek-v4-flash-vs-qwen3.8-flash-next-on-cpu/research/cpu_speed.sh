#!/bin/sh
# Pure-CPU throughput for both models. Uses the genuinely CPU-only binary:
# a CUDA build with -ngl 0 still offloads batch matmuls and is not a CPU number.
# -t 24 unpinned, per results/affinity.jsonl (every pinned config was slower).
BENCH="D:/Claude AI/AI Model Performance/qwen38-flash-next-cpu/research/build-cpu/bin/Release/llama-bench.exe"
FN="D:/Claude AI/AI Model Performance/qwen38-flash-next-cpu/models/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
DS="D:/Claude AI/AI Model Performance/deepseek-v4-vs-flash-next-1bit/models/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf"
OUT="D:/Claude AI/AI Model Performance/deepseek-v4-vs-flash-next-1bit/results/cpu_speed.jsonl"
run() {
  echo "### $1" >&2
  "$BENCH" -m "$2" -p 512 -n 128 -r 2 -t 24 -o jsonl \
    | python -c "import sys,json
for l in sys.stdin:
    l=l.strip()
    if not l.startswith('{'): continue
    r=json.loads(l); r['label']='$1'
    print(json.dumps(r))" >> "$OUT"
}
run flashnext-cpu "$FN"
run dsv4-cpu      "$DS"
echo "DONE" >&2

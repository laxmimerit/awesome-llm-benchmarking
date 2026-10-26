#!/bin/sh
# Does pinning to a homogeneous core set beat 24 unpinned threads?
# Windows has no taskset; llama.cpp does this natively with -C/--cpu-strict.
# Core map from GetLogicalProcessorInformationEx on this 285K (interleaved):
#   P-cores (8):  cpus 0,1,10,11,12,13,22,23   mask C03C03
#   E-cores (16): cpus 2-9,14-21               mask 3FC3FC
BENCH="D:/Claude AI/AI Model Performance/qwen38-flash-next-cpu/research/build-cpu/bin/Release/llama-bench.exe"
M="D:/Claude AI/AI Model Performance/qwen38-flash-next-cpu/models/UD-IQ1_S/Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"
OUT="D:/Claude AI/AI Model Performance/deepseek-v4-vs-flash-next-1bit/results/affinity.jsonl"
run() {
  echo "### $1  (-t $2 -C $3 --cpu-strict $4)" >&2
  "$BENCH" -m "$M" -p 512 -n 128 -r 3 -t "$2" -C "$3" --cpu-strict "$4" -o jsonl \
    | python -c "import sys,json
for l in sys.stdin:
    l=l.strip()
    if not l.startswith('{'): continue
    r=json.loads(l); r['affinity_label']='$1'
    print(json.dumps(r))" >> "$OUT"
}
run all24-unpinned 24 0x0     0
run all24-pinned   24 FFFFFF  1
run pcores-only     8 C03C03  1
run ecores-only    16 3FC3FC  1
echo "DONE" >&2

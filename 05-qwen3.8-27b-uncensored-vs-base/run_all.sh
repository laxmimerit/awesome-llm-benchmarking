set -u
cd "D:/Claude AI/AI Model Performance/qwen38-uncensored-vs-base"
echo "=== waiting for download $(date +%H:%M:%S)"
while ! grep -q ALLDONE logs/dl.log 2>/dev/null; do sleep 20; done
SZ=$(stat -c %s models/Qwen3.8-27B-Uncensored-Q4_K_M.gguf)
echo "=== download done, size=$SZ (expect 16810714496) $(date +%H:%M:%S)"
if [ "$SZ" != "16810714496" ]; then echo "SIZE MISMATCH, aborting"; exit 1; fi
echo "=== INFERENCE BENCH $(date +%H:%M:%S)"
python bench.py quality
python bench.py safety
python bench.py speed
python bench.py ppl
echo "=== WEIGHT DIFF $(date +%H:%M:%S)"
python ggufdiff.py structure
python ggufdiff.py weights
python ggufdiff.py direction
echo "=== REPORT $(date +%H:%M:%S)"
python report.py --md >/dev/null 2>&1 || python report.py
echo "=== PIPELINE COMPLETE $(date +%H:%M:%S)"

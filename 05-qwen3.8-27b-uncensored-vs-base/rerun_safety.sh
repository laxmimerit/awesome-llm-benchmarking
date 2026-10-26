set -u
cd "D:/Claude AI/AI Model Performance/qwen38-uncensored-vs-base"
echo "=== waiting for main pipeline to finish (GPU busy) $(date +%H:%M:%S)"
while ! grep -qE "PIPELINE COMPLETE" logs/run_all.log 2>/dev/null; do sleep 15; done
echo "=== pipeline done; backing up old safety data $(date +%H:%M:%S)"
cp results.jsonl results.pre_safety_fix.jsonl
# drop the buggy safety rows, keep quality/speed/ppl
python - <<'PY'
import json
keep=[l for l in open('results.jsonl',encoding='utf-8')
      if json.loads(l).get('stage')!='safety']
open('results.jsonl','w',encoding='utf-8').writelines(keep)
print('kept',len(keep),'non-safety rows')
PY
mv transcripts transcripts_buggy_2048 2>/dev/null || true
echo "=== re-running safety with 16k budget $(date +%H:%M:%S)"
python bench.py safety
echo "=== rebuilding report $(date +%H:%M:%S)"
python report.py >/dev/null 2>&1 || python report.py
echo "=== SAFETY RERUN COMPLETE $(date +%H:%M:%S)"

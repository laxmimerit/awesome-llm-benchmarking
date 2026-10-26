set -u
cd "D:/Claude AI/AI Model Performance/qwen38-uncensored-vs-base"
echo "=== waiting for thinking-mode safety rerun to finish $(date +%H:%M:%S)"
while ! grep -qE "SAFETY RERUN COMPLETE" logs/rerun_safety.log 2>/dev/null; do sleep 15; done
echo "=== running no-think safety pass $(date +%H:%M:%S)"
python bench.py safety_nothink
echo "=== rebuilding report $(date +%H:%M:%S)"
python report.py >/dev/null 2>&1 || python report.py
echo "=== NOTHINK COMPLETE $(date +%H:%M:%S)"

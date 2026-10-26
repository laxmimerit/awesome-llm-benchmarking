"""Check whether both runtimes recorded the fields needed for a fair,
end-to-end comparison without re-running anything.

completion_tokens / wall_s is measured identically on both sides: total tokens
the server says it produced, over total wall-clock time the client waited. It
includes streaming overhead and hidden thinking for both, so it sidesteps the
mismatch between llama.cpp's stream-timed decode and Ollama's internal
eval_duration.
"""

import json

d = json.load(open("results.json", encoding="utf-8"))

for key in (
    "ollama-ctx32k-kvf16-faon-specnone-think",
    "llamacpp-ctx32k-kvf16-faon-specnone-think",
):
    v = d.get(key)
    if not v:
        print(f"{key}: absent\n")
        continue
    runs = [
        r for r in v.get("runs", [])
        if not r.get("error") and r.get("kind") == "decode"
    ]
    print(f"=== {key}  ({len(runs)} decode runs) ===")
    for r in runs[:3]:
        ct, wall = r.get("completion_tokens"), r.get("wall_s")
        e2e = round(ct / wall, 2) if ct and wall else None
        print(
            f"  {r['task']:<18} completion_tok={ct:<6} wall_s={wall:<9} "
            f"end_to_end_tps={e2e:<8} reported_decode_tps={r.get('decode_tps')}"
        )
    print()

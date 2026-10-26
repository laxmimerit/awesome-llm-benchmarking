"""Show per-repeat decode speeds for one config, to explain a wide spread."""

import json
import sys

key_sub = sys.argv[1] if len(sys.argv) > 1 else "ngram"
d = json.load(open("results.json", encoding="utf-8"))

for k, v in d.items():
    if k.startswith("_") or key_sub not in k:
        continue
    print(f"\n=== {k} ===")
    runs = [r for r in v.get("runs", []) if not r.get("error")]
    tasks = sorted({r["task"] for r in runs})
    for t in tasks:
        rs = sorted(
            [r for r in runs if r["task"] == t], key=lambda r: r.get("rep", 0)
        )
        cells = " ".join(
            f"rep{r.get('rep')}={r.get('decode_tps')}" for r in rs
        )
        print(f"  {t:<20} {cells}")

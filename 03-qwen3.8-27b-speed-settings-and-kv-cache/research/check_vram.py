"""Check raw (unsubtracted) VRAM to see whether the single global idle
baseline is drifting under us as background desktop usage changes.
"""

import json

d = json.load(open("results.json", encoding="utf-8"))
base = d["_meta"]["baseline_vram_mib"]
print(f"global baseline recorded once at start: {base} MiB\n")

for k, v in d.items():
    if k.startswith("_") or "config" not in v:
        continue
    runs = [r for r in v.get("runs", []) if not r.get("error")]
    raw = [r["vram_peak_raw_mib"] for r in runs if r.get("vram_peak_raw_mib")]
    print(k)
    print(
        f"   raw_at_load={v.get('vram_resident_raw_mib')}"
        f"  raw_peak_min={min(raw) if raw else None}"
        f"  raw_peak_max={max(raw) if raw else None}"
    )
    print(f"   vram_after_release={v.get('vram_after_release_mib')}")

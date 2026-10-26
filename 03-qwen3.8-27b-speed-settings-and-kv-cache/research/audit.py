"""Audit every collected config for problems before anything is published.

Flags, per config:
  * errored runs
  * wide repeat spread (machine was not quiet, or a cache warmed mid-run)
  * missing per-process VRAM (measured before that fix landed)
  * missing draft stats where speculation was enabled
  * timing source, so cross-runtime comparisons are not made accidentally
  * quality below 100%

Read-only: touches results.json and nothing else. No GPU use.
"""

import json
import statistics

import summarize as S

data = S.load()
rows = {r["id"]: r for r in S.rows(data) if "FAILED" not in r}

SPREAD_WARN = 10.0

problems = {}


def add(cfg_id, issue):
    problems.setdefault(cfg_id, []).append(issue)


for key, entry in data.items():
    if key.startswith("_") or "config" not in entry:
        continue
    cfg = entry["config"]
    runs = entry.get("runs", [])
    ok = [r for r in runs if not r.get("error")]
    bad = [r for r in runs if r.get("error")]
    row = rows.get(key)

    if bad:
        kinds = sorted({r.get("kind", "?") for r in bad})
        add(key, f"{len(bad)} errored runs (kinds: {', '.join(kinds)})")

    if row:
        sp = row.get("decode_spread_pct")
        if sp is not None and sp > SPREAD_WARN:
            add(key, f"repeat spread {sp}% (>{SPREAD_WARN}% = not a quiet run)")
        if row.get("quality_pct") is not None and row["quality_pct"] < 100:
            add(key, f"quality {row['quality_pct']}%")
        if not row.get("vram_proc_mib"):
            add(key, "no per-process VRAM (predates that fix)")
        if cfg.get("spec", "none") != "none" and not row.get("accept_rate"):
            add(key, "speculation enabled but no acceptance stats")

    # Which client measured this? Mixing the two invalidates comparisons.
    srcs = {r.get("timing_source", "stream-wallclock") for r in ok}
    if len(srcs) > 1:
        add(key, f"mixed timing sources: {srcs}")

    if not ok:
        add(key, "no successful runs")


print("=" * 78)
print("CONFIG AUDIT")
print("=" * 78)
print(f"total configs: {len(rows)}")
print(f"configs with problems: {len(problems)}\n")

for cfg_id in sorted(problems):
    print(f"[!] {cfg_id}")
    for p in problems[cfg_id]:
        print(f"      - {p}")
print()

# ---- cross-runtime timing parity -------------------------------------------
print("=" * 78)
print("TIMING SOURCE BY RUNTIME")
print("=" * 78)
by_src = {}
for key, entry in data.items():
    if key.startswith("_") or "config" not in entry:
        continue
    rt = entry["config"]["runtime"]
    for r in entry.get("runs", []):
        if r.get("error"):
            continue
        src = r.get("timing_source", "stream-wallclock")
        by_src.setdefault((rt, src), 0)
        by_src[(rt, src)] += 1
for (rt, src), n in sorted(by_src.items()):
    print(f"  {rt:<10} {src:<20} {n:>5} runs")
print(
    "\n  NOTE: decode_tps is not comparable across these two sources.\n"
    "  Use completion_tokens / wall_s, which is measured identically."
)

# ---- coverage ---------------------------------------------------------------
print("\n" + "=" * 78)
print("COVERAGE BY AXIS")
print("=" * 78)
axes = {
    "spec": lambda r: f"{r['spec']}"
    + (f" n={r['n_max']}" if r["n_max"] else ""),
    "kv": lambda r: r["kv"],
    "ctx_k": lambda r: f"{r['ctx_k']}K",
    "fa": lambda r: r["fa"],
    "think/effort": lambda r: (
        "off" if not r["think"] else (r["effort"] or "default")
    ),
    "runtime": lambda r: r["runtime"],
}
for name, fn in axes.items():
    vals = sorted({fn(r) for r in rows.values()})
    print(f"  {name:<14} {', '.join(vals)}")

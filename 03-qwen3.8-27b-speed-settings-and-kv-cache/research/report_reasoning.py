"""Focused view of the thinking / reasoning-effort axis.

Shows what each reasoning level actually spends and what it buys, since the
levels are a system-prompt nudge rather than a hard budget.
"""

import summarize as S

rows = [r for r in S.rows(S.load()) if "FAILED" not in r]
rows = [r for r in rows if r["ctx_k"] == 32 and r["kv"] == "f16"]

hdr = (
    f"{'mode':<22}{'spec':<16}{'think tok':<11}{'ttft s':<9}"
    f"{'ttfa s':<9}{'decode tps':<12}{'accept':<9}{'quality':<9}"
)
print(hdr)
print("-" * len(hdr))

for r in sorted(rows, key=lambda r: (r["spec"], str(r["effort"]), r["think"])):
    mode = "thinking off" if not r["think"] else f"think {r['effort'] or 'default'}"
    spec = r["spec"] + (f" n={r['n_max']}" if r["n_max"] else "")
    print(
        f"{mode:<22}{spec:<16}"
        f"{str(r['thinking_chunks'] or 0):<11}"
        f"{str(r['ttft_s']):<9}{str(r['ttfa_s']):<9}"
        f"{str(r['decode_tps']):<12}{str(r['accept_rate'] or ''):<9}"
        f"{str(r['quality_pct']):<9}"
    )

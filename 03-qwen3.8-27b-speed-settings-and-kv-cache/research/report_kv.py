"""Focused view of the KV-cache precision axis at 32K context.

The question this answers: how much memory does quantising the KV cache
actually save, and does it cost accuracy? Long-context recall (the needle
tasks) is where quantisation damage would show up first, so it is broken out.
"""

import summarize as S

rows = [r for r in S.rows(S.load()) if "FAILED" not in r]
rows = [
    r for r in rows
    if r["ctx_k"] == 32 and r["fa"] == "on" and r["think"]
    and r["effort"] is None and r["runtime"] == "llamacpp"
    and r["spec"] in ("none", "draft-mtp")
    and (r["n_max"] in (None, 2))
]

hdr = (
    f"{'KV type':<10}{'spec':<12}{'decode tps':<12}{'VRAM MB':<11}"
    f"{'saved vs f16':<14}{'accept':<9}{'quality':<9}{'needles':<26}{'spread%':<8}"
)
print(hdr)
print("-" * len(hdr))

f16 = {
    r["spec"]: r["vram_proc_mib"]
    for r in rows if r["kv"] == "f16"
}

for kv in ("f16", "q8_0", "q4_0"):
    for spec in ("none", "draft-mtp"):
        r = next(
            (x for x in rows if x["kv"] == kv and x["spec"] == spec), None
        )
        if not r:
            continue
        base = f16.get(spec)
        saved = (
            f"{base - r['vram_proc_mib']:.0f}" if base and kv != "f16" else "-"
        )
        needles = ",".join(
            f"{k.replace('needle_d','d')}={'ok' if v else 'FAIL'}"
            for k, v in sorted(r["needle"].items())
        )
        print(
            f"{kv:<10}{spec:<12}{r['decode_tps']:<12}"
            f"{r['vram_proc_mib']:<11.0f}{saved:<14}"
            f"{str(r['accept_rate'] or '-'):<9}{r['quality_pct']:<9}"
            f"{needles:<26}{str(r['decode_spread_pct']):<8}"
        )

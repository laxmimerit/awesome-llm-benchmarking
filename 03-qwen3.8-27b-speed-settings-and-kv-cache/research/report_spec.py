"""The speculative-decoding ladder: what each draft depth buys and costs.

All rows are at the same context, cache precision and reasoning setting, so
the only thing varying is --spec-draft-n-max.
"""

import summarize as S

rows = [
    r for r in S.rows(S.load())
    if "FAILED" not in r
    and r["runtime"] == "llamacpp" and r["ctx_k"] == 32 and r["kv"] == "f16"
    and r["fa"] == "on" and r["think"] is True and r["effort"] is None
]


def order(r):
    if r["spec"] == "none":
        return (0, 0)
    if r["spec"] == "draft-mtp":
        return (1, r["n_max"] or 0)
    return (2, 0)


rows = sorted(rows, key=order)
base = next((r["decode_tps"] for r in rows if r["spec"] == "none"), None)

hdr = (
    f"{'setting':<16}{'decode tps':<12}{'vs off':<9}{'accept':<9}"
    f"{'accepted len':<14}{'prefill tps':<13}{'VRAM MiB':<11}"
    f"{'tok/Wh':<9}{'quality':<9}{'spread%':<8}"
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    if r["spec"] == "none":
        name = "MTP off"
    elif r["spec"] == "draft-mtp":
        name = f"draft-mtp n={r['n_max']}"
    else:
        name = r["spec"]
    speed = r["decode_tps"]
    ratio = f"{speed / base:.2f}x" if base and r["spec"] != "none" else "-"
    print(
        f"{name:<16}{speed:<12}{ratio:<9}"
        f"{str(r['accept_rate'] or '-'):<9}"
        f"{str(r['mean_accepted_len'] or '-'):<14}"
        f"{str(r['prefill_tps']):<13}{r['vram_proc_mib']:<11.0f}"
        f"{str(r['tokens_per_wh']):<9}{r['quality_pct']:<9}"
        f"{str(r['decode_spread_pct']):<8}"
    )

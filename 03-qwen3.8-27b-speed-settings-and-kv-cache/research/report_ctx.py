"""Context scaling: what a bigger window costs to configure vs to actually use.

Two different questions get confused in most benchmarks:
  * raising -c        -> costs VRAM, barely costs decode speed
  * filling the window -> costs a great deal of latency, because attention is
                          quadratic in prompt length
Both are reported side by side here.
"""

import summarize as S




def wanted_kv(ctx_k):
    """The KV type the ctx sweep used at each size: f16 while it fits."""
    return "f16" if ctx_k <= 64 else "q8_0"


rows = [
    r for r in S.rows(S.load())
    if "FAILED" not in r
    and r["spec"] == "draft-mtp" and r["n_max"] == 2
    and r["effort"] is None and r["runtime"] == "llamacpp" and r["fa"] == "on"
    # Other stages also produced 32K configs with different cache precision
    # and with thinking disabled. Every axis except context must be pinned or
    # the curve silently compares different things at different points.
    and r["kv"] == wanted_kv(r["ctx_k"])
    and r["think"] is True
]
rows = sorted({r["ctx_k"]: r for r in rows}.values(), key=lambda r: r["ctx_k"])

hdr = (
    f"{'ctx':>6}{'KV':>7}{'VRAM GB':>10}"
    f"{'short-prompt decode':>21}"
    f"{'full prompt':>13}{'TTFT':>10}{'decode':>10}{'quality':>9}"
)
print(hdr)
print("-" * len(hdr))
for r in rows:
    vram = r["vram_proc_mib"] / 1024 if r["vram_proc_mib"] else 0
    ttft = r.get("long_ttft_s")
    print(
        f"{r['ctx_k']:>5}K{r['kv']:>7}{vram:>10.1f}"
        f"{r['decode_tps']:>21}"
        f"{str(r.get('long_prompt_tokens') or '-'):>13}"
        f"{(f'{ttft:.0f}s' if ttft else '-'):>10}"
        f"{str(r.get('long_decode_tps') or '-'):>10}"
        f"{r['quality_pct']:>9}"
    )

"""Cross-check the numbers written in FINDINGS.md against results.json.

Tables were hand-written across several sessions while re-runs were updating
shared configs, so a stale figure is easy to leave behind. This re-derives the
headline values and prints them for comparison.
"""

import summarize as S

ROWS = [r for r in S.rows(S.load()) if "FAILED" not in r]


def get(**crit):
    return S.pick(ROWS, **crit)


BASE = {}

print("=== values FINDINGS.md should quote ===\n")

print("-- ladder (32K, f16, thinking on) --")
b = get(**BASE, kv="f16", spec="none", think=True)
print(f"  MTP off        decode={b['decode_tps']}  vram={b['vram_proc_mib']:.0f}  tok/Wh={b['tokens_per_wh']}")
for n in (1, 2, 3, 4, 5):
    r = get(**BASE, kv="f16", spec="draft-mtp", n_max=n, think=True)
    if r:
        print(
            f"  n={n}            decode={r['decode_tps']}  "
            f"{r['decode_tps']/b['decode_tps']:.2f}x  accept={r['accept_rate']}  "
            f"len={r['mean_accepted_len']}  vram={r['vram_proc_mib']:.0f}  "
            f"tok/Wh={r['tokens_per_wh']}"
        )
g = get(**BASE, kv="f16", spec="ngram-mod", think=True)
if g:
    print(f"  ngram-mod      decode={g['decode_tps']}  {g['decode_tps']/b['decode_tps']:.2f}x  "
          f"cold={g['decode_tps_cold']}  len={g['mean_accepted_len']}")

print("\n-- thinking / reasoning --")
for label, crit in [
    ("think on,  no spec", dict(kv="f16", spec="none", think=True)),
    ("think off, no spec", dict(kv="f16", spec="none", think=False)),
    ("think on,  n=3", dict(kv="f16", spec="draft-mtp", n_max=3, think=True)),
    ("think off, n=2", dict(kv="f16", spec="draft-mtp", n_max=2, think=False)),
    ("xhigh, n=2", dict(kv="f16", spec="draft-mtp", n_max=2, think=True, effort="xhigh")),
    ("medium, n=2", dict(kv="f16", spec="draft-mtp", n_max=2, think=True, effort="medium")),
    ("low, n=2", dict(kv="f16", spec="draft-mtp", n_max=2, think=True, effort="low")),
]:
    c = dict(BASE)
    c.update(crit)
    if "effort" in crit:
        c["effort"] = crit["effort"]
    r = get(**c)
    if r:
        print(
            f"  {label:<20} think_tok={str(r['thinking_chunks']):<7} "
            f"ttfa={str(r['ttfa_s']):<7} decode={str(r['decode_tps']):<8} "
            f"accept={str(r['accept_rate'] or '-'):<8} quality={r['quality_pct']}"
        )

print("\n-- ngram cold start (first pass, before its cache warms) --")
for spec in ("none", "ngram-mod"):
    r = get(**BASE, kv="f16", spec=spec, think=True)
    if r:
        print(
            f"  {spec:<12} median={str(r['decode_tps']):<9} "
            f"cold_rep0={str(r['decode_tps_cold']):<9}"
        )
nb = get(**BASE, kv="f16", spec="none", think=True)
ng = get(**BASE, kv="f16", spec="ngram-mod", think=True)
if nb and ng and nb.get("decode_tps_cold") and ng.get("decode_tps_cold"):
    print(
        f"  -> cold ratio: {ng['decode_tps_cold'] / nb['decode_tps_cold']:.2f}x"
    )

print("\n-- flash attention --")
for fa in ("on", "off"):
    r = get(runtime="llamacpp", ctx_k=32, kv="f16", fa=fa, spec="none",
            think=True, effort=None)
    if r:
        print(
            f"  fa={fa:<4} decode={str(r['decode_tps']):<8} "
            f"prefill={str(r['prefill_tps']):<9} ttft={str(r['ttft_s']):<7} "
            f"vram={r['vram_proc_mib']:.0f}"
        )

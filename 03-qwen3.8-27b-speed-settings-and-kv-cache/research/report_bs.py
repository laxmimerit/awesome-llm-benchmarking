"""--spec-draft-backend-sampling, on vs off, at two draft lengths.

Ollama enables this flag by default; upstream llama.cpp does not. It moves
draft-token sampling onto the GPU backend instead of round-tripping to the
host, so the effect should show up in decode speed without moving the
acceptance rate -- acceptance is a property of what the draft head predicts,
not of where the sampling runs. A shifted acceptance rate would mean the flag
changes the sampling result, not just its location, which is worth knowing.
"""

import summarize as S

ROWS = [r for r in S.rows(S.load()) if "FAILED" not in r]


def get(**crit):
    return S.pick(ROWS, **crit)


BASE = dict(spec="draft-mtp")

print("=== backend sampling (32K, f16, thinking on) ===\n")
print(f"{'draft n':<9}{'bsampling':<12}{'decode':<10}{'accept':<9}"
      f"{'mean len':<10}{'tok/Wh':<9}{'vram MB':<10}")
print("-" * 69)

for n in (3, 4):
    base = None
    for bs in (True, False):
        r = get(**BASE, n_max=n, backend_sampling=bs)
        if not r:
            print(f"{n:<9}{str(bs):<12}(not measured)")
            continue
        if bs:
            base = r
        delta = ""
        if base and not bs and base["decode_tps"]:
            delta = f"  ({r['decode_tps'] / base['decode_tps']:.3f}x vs on)"
        print(
            f"{n:<9}{('on' if bs else 'off'):<12}"
            f"{str(r['decode_tps']):<10}{str(r['accept_rate']):<9}"
            f"{str(r['mean_accepted_len']):<10}{str(r['tokens_per_wh']):<9}"
            f"{r['vram_proc_mib']:<10.0f}{delta}"
        )
    print()

# The unflagged rows from the original ladder used the build default (off).
print("-- for reference, the original ladder rows (flag left unset) --")
for n in (3, 4):
    r = get(**BASE, n_max=n, backend_sampling=None)
    if r:
        print(f"  n={n}  decode={r['decode_tps']}  accept={r['accept_rate']}  "
              f"len={r['mean_accepted_len']}")

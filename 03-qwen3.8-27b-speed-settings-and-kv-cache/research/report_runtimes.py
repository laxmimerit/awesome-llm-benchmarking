"""How the collected configs split across runtimes."""

import summarize as S

rows = [r for r in S.rows(S.load()) if "FAILED" not in r]

for rt in ("llamacpp", "ollama"):
    subset = [r for r in rows if r["runtime"] == rt]
    print(f"\n=== {rt}: {len(subset)} configs ===")
    for r in sorted(subset, key=lambda x: x["id"]):
        spec = r["spec"] + (f" n={r['n_max']}" if r["n_max"] else "")
        mode = "off" if not r["think"] else (r["effort"] or "default")
        err = f"  [{r['errors']} errored runs]" if r.get("errors") else ""
        print(
            f"  ctx={r['ctx_k']:>4}K  kv={r['kv']:<5} fa={r['fa']:<4} "
            f"spec={spec:<14} think={mode:<8} "
            f"decode={str(r['decode_tps']):<8}{err}"
        )

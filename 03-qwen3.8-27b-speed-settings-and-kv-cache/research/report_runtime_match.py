"""Ollama vs llama.cpp, with the defaults gap separated from the runtime gap.

The naive version of this comparison -- Ollama out of the box against
llama-server out of the box -- reports Ollama as roughly twice as fast. That
number is real but it does not measure what it appears to measure. Ollama does
not run stock llama.cpp: probe_ollama_flags.py read its child process's command
line and found it enables the MTP draft head at n=4 with backend sampling, and
overrides the batch sizes. The "runtime" gap was a flags gap.

Three rows per kv setting separate the two effects:

    ollama              what Ollama does by default
    llamacpp + match    llama-server given Ollama's exact flags
    llamacpp stock      llama-server on its own defaults

    ollama vs match  -> the actual runtime overhead
    match vs stock   -> what the defaults are worth
"""

import summarize as S

rows = {r["id"]: r for r in S.rows(S.load()) if "FAILED" not in r}


def get(**crit):
    for r in rows.values():
        if all(r.get(k) == v for k, v in crit.items()):
            return r
    return None


def fmt(r):
    if not r:
        return None
    return r


print(__doc__.split("Three rows")[0].strip())
print("\n" + "=" * 78)

for kv in ("f16", "q8_0"):
    print(f"\n### kv = {kv}\n")
    common = dict(ctx_k=32, fa="on", think=True, effort=None)

    ol = get(runtime="ollama", kv=kv, **common)
    match = get(runtime="llamacpp", kv=kv, spec="draft-mtp", n_max=4,
                backend_sampling=True, batch=1024, ubatch=1024, **common)
    stock = get(runtime="llamacpp", kv=kv, spec="none", backend_sampling=None,
                batch=None, **common)

    print(f"{'config':<22}{'decode':<10}{'prefill':<11}{'ttft':<9}"
          f"{'vram MB':<11}{'accept':<9}{'errors':<7}")
    print("-" * 79)
    for label, r in (("ollama (default)", ol),
                     ("llamacpp + ollama flags", match),
                     ("llamacpp (stock)", stock)):
        if not r:
            print(f"{label:<22}(not measured)")
            continue
        print(
            f"{label:<22}{str(r['decode_tps']):<10}{str(r['prefill_tps']):<11}"
            f"{str(r['ttft_s']):<9}{r['vram_proc_mib']:<11.0f}"
            f"{str(r['accept_rate'] or '-'):<9}{str(r.get('errors') or 0):<7}"
        )

    print()
    if ol and match and match["decode_tps"]:
        d = ol["decode_tps"] / match["decode_tps"]
        print(f"  runtime overhead   ollama / llamacpp-matched = {d:.3f}x"
              f"   ({'ollama ahead' if d > 1 else 'llama.cpp ahead'})")
    if match and stock and stock["decode_tps"]:
        d = match["decode_tps"] / stock["decode_tps"]
        print(f"  value of defaults  matched / stock          = {d:.3f}x")
    if ol and stock and stock["decode_tps"]:
        d = ol["decode_tps"] / stock["decode_tps"]
        print(f"  naive headline     ollama / stock            = {d:.3f}x"
              f"   <- what a careless benchmark would report")

print("\n" + "=" * 78)
print("""
Note on timing source: Ollama rows are timed from its native /api/chat
eval_count / eval_duration, llama.cpp rows from the SSE stream. Both count
reasoning tokens; an earlier version of this harness read only `content` from
the OpenAI-compatible endpoint, which hid Qwen3.8's thinking output and
reported an impossible 3477 tok/s.

Note on chat template: Ollama launches with `--no-jinja --chat-template chatml`
rather than the model's own Jinja template, so the two runtimes are not
prompting the model identically even when every performance flag matches.
""".strip())

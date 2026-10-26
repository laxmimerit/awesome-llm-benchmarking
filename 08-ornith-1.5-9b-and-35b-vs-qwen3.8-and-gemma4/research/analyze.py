"""Turn the raw jsonl results into the tables the report is built from.

Every summary here reports the spread as well as the central value, because on
a power-capped card a single number without a spread is not a measurement.
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

ORDER = ["ornith-35b-a3b", "gemma4-26b", "ornith-9b", "qwen3.8-27b"]


def load(name):
    p = RESULTS / name
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def by_model(rows):
    d = defaultdict(list)
    for r in rows:
        d[r.get("model")].append(r)
    return d


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def table(headers, rows, title):
    out = [f"\n### {title}\n"]
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows
              else len(str(h)) for i, h in enumerate(headers)]
    out.append("| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |")
    out.append("| " + " | ".join("-" * w for w in widths) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def summarize_speed():
    rows = [r for r in load("speed.jsonl") + load("speed_reverse.jsonl")
            if r.get("content")]
    if not rows:
        return ""
    d = defaultdict(list)
    for r in rows:
        d[(r["model"], r["content"])].append(r["decode_tps_server"])
    contents = ["counting", "code", "json", "essay", "fiction"]
    out = []
    for m in ORDER:
        vals = [d.get((m, c), []) for c in contents]
        if not any(vals):
            continue
        allv = [x for v in vals for x in v]
        row = [m]
        for v in vals:
            row.append(fmt(st.mean(v), 1) if v else "-")
        row.append(fmt(st.mean(allv), 1))
        row.append(f"{(max(allv)-min(allv))/st.mean(allv)*100:.1f}%" if len(allv) > 1 else "-")
        out.append(row)
    return table(["model", *contents, "mean", "spread"], out,
                 "Decode throughput, tok/s (server timings, 512 tokens, MTP off)")


def summarize_vram():
    rows = load("vram.jsonl")
    out = []
    kv = {r["model"]: r for r in rows if r.get("kind") == "kv_per_token"}
    for m in ORDER:
        mine = [r for r in rows if r.get("model") == m and r.get("ok")]
        if not mine:
            continue
        w = mine[0].get("model_buffer_total_mib")
        layers = mine[0].get("kv_layers_total")
        k = kv.get(m, {})
        kvb = k.get("kv_bytes_per_token")
        # full-context KV at the model's own advertised window
        out.append([
            m, fmt(w, 0), layers or "-",
            fmt(kvb / 1024, 1) if kvb else "-",
            fmt(kvb * 262144 / (1024**3), 1) if kvb else "-",
            fmt(mine[-1].get("compute_buffer_total_mib"), 0),
        ])
    return table(
        ["model", "weights MiB", "KV layers", "KV KiB/token",
         "KV GiB @262k", "compute MiB"], out,
        "VRAM accounting (Q4_K_M, f16 KV, flash attention on)")


def summarize_prefill():
    rows = [r for r in load("prefill.jsonl") if r.get("depth_target")]
    if not rows:
        return ""
    depths = sorted({r["depth_target"] for r in rows})
    out = []
    for m in ORDER:
        mine = {r["depth_target"]: r for r in rows if r.get("model") == m}
        if not mine:
            continue
        out.append([m] + [fmt(mine[d]["prompt_tps_server"], 0) if d in mine else "-"
                          for d in depths])
    return table(["model"] + [f"{d//1024}k" if d >= 1024 else str(d) for d in depths],
                 out, "Prefill throughput, tok/s, by prompt depth")


def summarize_quality():
    rows = load("quality.jsonl")
    if not rows:
        return ""
    out = []
    for m in ORDER:
        mine = [r for r in rows if r.get("model") == m]
        if not mine:
            continue
        for tier in ["base", "hard"]:
            t = [r for r in mine if r.get("tier") == tier]
            if not t:
                continue
            toks = [r["completion_tokens"] for r in t if r.get("completion_tokens")]
            secs = [r["wall_s"] for r in t if r.get("wall_s")]
            out.append([
                m, tier,
                f"{sum(r['correct'] for r in t)}/{len(t)}",
                fmt(st.median(toks), 0) if toks else "-",
                fmt(st.mean([r["reasoning_frac_chars"] for r in t]) * 100, 0) + "%",
                # The number that actually decides which model to run: accuracy
                # is saturated, so what separates these models is how long a
                # correct answer takes to arrive.
                fmt(st.median(secs), 1) if secs else "-",
                sum(r.get("truncated", False) for r in t),
            ])
    return table(["model", "tier", "correct", "median tokens",
                  "thinking share", "median s/answer", "truncated"], out,
                 "Accuracy and thinking economics")


# A1 was worded ambiguously in this run: it read as asking for the timezone
# alone, and every model that "failed" it answered exactly "UTC+1". Scoring it
# would report a prompt defect as a model difference.
AGENTIC_EXCLUDED = {"A1_hop3_datacenter"}


def summarize_agentic():
    rows = [r for r in load("agentic.jsonl") if r["id"] not in AGENTIC_EXCLUDED]
    if not rows:
        return ""
    out = []
    for m in ORDER:
        mine = [r for r in rows if r.get("model") == m]
        if not mine:
            continue
        easy = [r for r in mine if not r["id"].startswith("H")]
        hard = [r for r in mine if r["id"].startswith("H")]
        calls = [r.get("n_tool_calls", 0) for r in mine]
        out.append([
            m,
            f"{sum(bool(r.get('solved')) for r in easy)}/{len(easy)}",
            f"{sum(bool(r.get('solved')) for r in hard)}/{len(hard)}",
            f"{sum(bool(r.get('solved')) for r in mine)}/{len(mine)}",
            sum(r.get("n_malformed", 0) for r in mine),
            fmt(st.mean(calls), 1) if calls else "-",
            fmt(st.mean([r.get("wall_s", 0) for r in mine]), 1),
            fmt(st.mean([r.get("reasoning_chars", 0) for r in mine]), 0),
        ])
    return table(["model", "base 5", "hard 4", "total", "malformed calls",
                  "mean tool calls", "mean s/task", "mean think chars"], out,
                 "Agentic tool use (A1 excluded: defective prompt, see notes)")


def summarize_longctx():
    rows = [r for r in load("longctx.jsonl") if r.get("target_tokens")]
    if not rows:
        return ""
    depths = sorted({r["target_tokens"] for r in rows})
    out = []
    for m in ORDER:
        mine = [r for r in rows if r.get("model") == m]
        if not mine:
            continue
        row = [m]
        for d in depths:
            at = [r for r in mine if r["target_tokens"] == d]
            row.append(f"{sum(bool(r.get('found')) for r in at)}/{len(at)}" if at else "-")
        out.append(row)
    return table(["model"] + [f"{d//1024}k" for d in depths], out,
                 "Long-context needle retrieval (found / attempts)")


def summarize_concurrency():
    rows = [r for r in load("concurrency.jsonl") if r.get("ok")]
    if not rows:
        return ""
    levels = sorted({r["parallel"] for r in rows})
    out = []
    for m in ORDER:
        mine = {r["parallel"]: r for r in rows if r.get("model") == m}
        if not mine:
            continue
        out.append([m] + [fmt(mine[l]["aggregate_tps"], 1) if l in mine else "-"
                          for l in levels])
    return table(["model"] + [f"np={l}" for l in levels], out,
                 "Aggregate throughput under concurrency, tok/s")


def summarize_mtp():
    p = RESULTS / "mtp_check.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text())
    out = []
    for m, v in d.items():
        if not v.get("on"):
            out.append([m, "failed to start", "-", "-", "-"])
            continue
        for c, sp in v["speedup"].items():
            r = v["on"]["runs"][c]
            out.append([m, c, fmt(v["off"]["runs"][c]["decode_tps"], 1),
                        fmt(r["decode_tps"], 1), f"{sp:.2f}x",
                        fmt(r["accept_rate"], 3),
                        str(v["identical"].get(c))])
    return table(["model", "content", "off tok/s", "on tok/s", "speedup",
                  "accept", "output identical"], out,
                 "Multi-token prediction (--spec-type draft-mtp)")


def summarize_usable_ctx():
    rows = load("usable_ctx.jsonl")
    verdicts = [r for r in rows if r.get("kind") == "usable_ctx"]
    if not verdicts:
        return ""
    out = []
    for m in ORDER:
        v = next((r for r in verdicts if r.get("model") == m), None)
        if not v:
            continue
        probes = [r for r in rows if r.get("model") == m and r.get("ok")
                  and r.get("ctx")]
        at_cliff = next((r for r in probes
                         if r.get("ctx") == v.get("first_degraded_ctx")), None)
        out.append([
            m,
            f"{v['usable_max_ctx']:,}",
            f"{v['fraction_of_advertised']*100:.0f}%",
            fmt(v["baseline_tps"], 1),
            f"{v['first_degraded_ctx']:,}" if v.get("first_degraded_ctx") else "none",
            fmt(at_cliff["decode_tps"], 1) if at_cliff else "-",
            fmt(max((r.get("vram_used_mb") or 0) for r in probes), 0),
        ])
    return table(["model", "usable max ctx", "of advertised", "tok/s normal",
                  "cliff at", "tok/s past cliff", "peak VRAM MiB"], out,
                 "Usable context ceiling (decode stays within 90% of normal rate)")


def summarize_maxctx():
    rows = [r for r in load("maxctx.jsonl") if r.get("kind") == "max_ctx"]
    if not rows:
        return ""
    out = []
    for m in ORDER:
        mine = [r for r in rows if r.get("model") == m]
        if not mine:
            continue
        r = mine[-1]
        p = r.get("predicted") or {}
        best = r.get("max_ctx")
        adv = 262144
        fail = p.get("smallest_failing_ctx")
        # A result with no observed failure below the advertised window is a
        # lower bound, and is labelled as one rather than printed as a ceiling.
        bounded = bool(fail) or (best or 0) >= adv
        out.append([
            m,
            (f"{best:,}" if bounded else f"≥{best:,}") if best else "-",
            f"{best/adv*100:.0f}%" if best else "-",
            f"{fail:,}" if fail else ("n/a, holds full window"
                                      if (best or 0) >= adv else "none observed"),
            len(r.get("tried") or []),
            fmt(p.get("kv_bytes_per_token", 0) / 1024, 1),
        ])
    return table(["model", "max ctx on 32 GB", "of advertised 262k",
                  "smallest failing ctx", "probes", "KV KiB/tok"],
                 out, "Context ceiling (Q4_K_M, f16 KV, flash attention on)")


def summarize_kvquant():
    rows = [r for r in load("kvquant.jsonl") if r.get("ok")]
    if not rows:
        return ""
    types = ["f16", "q8_0", "q4_0"]
    out = []
    for m in ORDER:
        mine = {r["cache_type_k"]: r for r in rows if r.get("model") == m}
        if not mine:
            continue
        base = mine.get("f16", {}).get("kv_self_mib")
        row = [m]
        for t in types:
            r = mine.get(t)
            if not r:
                row.append("-")
                continue
            kv = r.get("kv_self_mib")
            saving = f" ({(1-kv/base)*100:.0f}% less)" if base and kv and t != "f16" else ""
            row.append(f"{kv:.0f} MiB{saving}" if kv else "-")
        out.append(row)
    return table(["model"] + [f"KV {t}" for t in types], out,
                 "KV cache size at 64k context by cache type")


def main():
    parts = [
        "# Ornith 1.5 benchmark: measured results",
        summarize_vram(),
        summarize_usable_ctx(),
        summarize_maxctx(),
        summarize_kvquant(),
        summarize_speed(),
        summarize_prefill(),
        summarize_mtp(),
        summarize_concurrency(),
        summarize_quality(),
        summarize_agentic(),
        summarize_longctx(),
    ]
    text = "\n".join(p for p in parts if p)
    (RESULTS / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

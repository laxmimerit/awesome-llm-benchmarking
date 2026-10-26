"""Aggregate results.json into per-config summary rows.

Speed is reported as the median across repeats, which is what the community
recipes use and what resists the occasional scheduler hiccup. Correctness is
reported as a pass count so a config that gets fast by getting dumb is visible
next to its speed number.

Usage:
    python summarize.py              # print all stages
    python summarize.py spec kv      # print selected stages
    python summarize.py --md         # emit results-summary.md
"""

import json
import os
import statistics
import sys

RESEARCH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(RESEARCH, "results.json")
SUMMARY_MD = os.path.join(RESEARCH, "results-summary.md")


def load():
    with open(RESULTS, encoding="utf-8") as f:
        return json.load(f)


def med(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.median(vals), 2) if vals else None


def _spread_pct(runs):
    """Worst per-task spread across repeats, as a % of that task's median.

    Each task is compared only against its own repeats, so genuinely
    different workloads (code vs prose) do not inflate the number. Anything
    above roughly 10% means the machine was not quiet during the run.
    """
    worst = None
    tasks = {r["task"] for r in runs}
    for t in tasks:
        vals = [
            r["decode_tps"] for r in runs
            if r["task"] == t and isinstance(r.get("decode_tps"), (int, float))
        ]
        if len(vals) < 2:
            continue
        m = statistics.median(vals)
        if not m:
            continue
        pct = 100.0 * (max(vals) - min(vals)) / m
        worst = pct if worst is None else max(worst, pct)
    return round(worst, 1) if worst is not None else None


def summarize_config(entry):
    """Collapse one config's runs into a single row of headline numbers."""
    cfg = entry["config"]
    runs = entry.get("runs", [])
    ok_runs = [r for r in runs if not r.get("error")]

    def by_kind(kind, field):
        return [r.get(field) for r in ok_runs if r.get("kind") == kind]

    decode = [r for r in ok_runs if r.get("kind") == "decode"]
    quality = [r for r in ok_runs if r.get("kind") in ("quality", "needle")]
    graded = [r for r in quality if r.get("correct") is not None]

    row = {
        "id": cfg["id"],
        "runtime": cfg["runtime"],
        "ctx_k": cfg["ctx"] // 1024,
        "kv": cfg["kv"],
        "fa": cfg["flash_attn"],
        "spec": cfg["spec"],
        "n_max": cfg.get("n_max"),
        "backend_sampling": cfg.get("backend_sampling"),
        "chat_template": cfg.get("chat_template"),
        "tag": cfg.get("tag"),
        "batch": cfg.get("batch"),
        "ubatch": cfg.get("ubatch"),
        "think": cfg.get("think", True),
        "effort": cfg.get("effort"),
        "load_s": entry.get("load_seconds"),
        # llama.cpp allocates transient buffers while loading and frees them
        # once warm, so the reading taken right after /health is not a stable
        # basis for comparison. What decides "will this fit on my card" is the
        # highest figure ever observed, so report that and keep the components
        # visible for transparency.
        "vram_at_load_mib": entry.get("vram_resident_mib"),
        # Whole-card figure, baseline-subtracted. Kept for reference only: it
        # still moves when the desktop allocates during a run.
        "vram_card_mib": max(
            [entry.get("vram_resident_mib") or 0]
            + [r.get("vram_peak_mib") or 0 for r in ok_runs]
        )
        or None,
        # Authoritative footprint: the server process's own dedicated memory,
        # which nothing else on the desktop can inflate.
        "vram_proc_mib": max(
            [entry.get("proc_vram_load_mib") or 0]
            + [r.get("proc_vram_peak_mib") or 0 for r in ok_runs]
        )
        or None,
        # Decode throughput is the number speculative decoding moves.
        "decode_tps": med([r.get("decode_tps") for r in decode]),
        # First-pass speed, before any cache has seen the prompt. n-gram
        # drafting learns from text it has already produced, so repeating an
        # identical prompt lets it replay from cache at absurd rates. Only the
        # cold figure reflects real use with varied prompts. MTP is stateless
        # per request, so for it the two agree.
        "decode_tps_cold": med(
            [r.get("decode_tps") for r in decode if r.get("rep") == 0]
        ),
        "decode_tps_all": med([r.get("decode_tps") for r in ok_runs]),
        "prefill_tps": med(by_kind("prefill", "prefill_tps")),
        # Latency when the context is genuinely full. `ttft_s` above is
        # measured with short prompts and so answers a different question:
        # what configuring a big window costs, not what filling it costs.
        "long_ttft_s": med(
            [r.get("ttft_s") for r in ok_runs if r.get("kind") == "needle"]
        ),
        "long_prompt_tokens": med(
            [r.get("prompt_tokens") for r in ok_runs if r.get("kind") == "needle"]
        ),
        "long_decode_tps": med(
            [r.get("decode_tps") for r in ok_runs if r.get("kind") == "needle"]
        ),
        "ttft_s": med([r.get("ttft_s") for r in decode]),
        "ttfa_s": med([r.get("ttfa_s") for r in ok_runs]),
        "power_mean_w": med([r.get("power_mean_w") for r in ok_runs]),
        "tokens_per_wh": med([r.get("tokens_per_wh") for r in decode]),
        "thinking_chunks": med([r.get("thinking_chunks") for r in ok_runs]),
        "accept_rate": (entry.get("draft_stats") or {}).get("accept_rate"),
        "mean_accepted_len": (entry.get("draft_stats") or {}).get(
            "mean_accepted_len"
        ),
        "baseline_mib": entry.get("baseline_vram_mib"),
        # Spread across repeats of the same workload. A machine that stalled
        # mid-run (sleep, thermal throttle, a background job stealing the GPU)
        # shows up here as a wide spread even though the median looks fine.
        "decode_spread_pct": _spread_pct(decode),
        "quality_pass": sum(1 for r in graded if r["correct"]),
        "quality_total": len(graded),
        "errors": len(runs) - len(ok_runs),
    }
    if row["quality_total"]:
        row["quality_pct"] = round(
            100.0 * row["quality_pass"] / row["quality_total"], 1
        )
    else:
        row["quality_pct"] = None

    # Per-task decode detail: acceptance is highly content dependent.
    row["decode_by_task"] = {
        t: med([r.get("decode_tps") for r in decode if r["task"] == t])
        for t in sorted({r["task"] for r in decode})
    }
    # Needle recall broken out by depth, to expose long-context quality loss.
    needles = [r for r in ok_runs if r.get("kind") == "needle"]
    row["needle"] = {
        t: all(r.get("correct") for r in needles if r["task"] == t)
        for t in sorted({r["task"] for r in needles})
    }
    return row


def rows(data):
    out = []
    for key, entry in data.items():
        if key.startswith("_") or not isinstance(entry, dict):
            continue
        if "config" not in entry:
            continue
        if entry.get("error"):
            out.append(
                {"id": key, "FAILED": entry["error"].splitlines()[0][:120]}
            )
            continue
        out.append(summarize_config(entry))
    return out


HEADERS = [
    ("id", 48),
    ("decode_tps", 12),
    ("decode_tps_cold", 16),
    ("accept_rate", 13),
    ("prefill_tps", 13),
    ("ttft_s", 9),
    ("vram_proc_mib", 15),
    ("quality_pct", 13),
    ("tokens_per_wh", 15),
    ("decode_spread_pct", 8),
]


def print_table(rs):
    head = "".join(h.ljust(w) for h, w in HEADERS)
    print(head)
    print("-" * len(head))
    for r in rs:
        if "FAILED" in r:
            print(f"{r['id'][:46].ljust(46)}FAILED: {r['FAILED']}")
            continue
        line = ""
        for h, w in HEADERS:
            v = r.get(h)
            line += (("" if v is None else str(v))).ljust(w)
        print(line)


def to_markdown(rs):
    cols = [
        ("id", "config"),
        ("decode_tps", "decode tok/s"),
        ("accept_rate", "MTP accept"),
        ("prefill_tps", "prefill tok/s"),
        ("ttft_s", "TTFT s"),
        ("vram_proc_mib", "model VRAM MiB"),
        ("quality_pct", "quality %"),
        ("tokens_per_wh", "tok/Wh"),
        ("load_s", "load s"),
    ]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |"]
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rs:
        if "FAILED" in r:
            continue
        lines.append(
            "| " + " | ".join(
                "" if r.get(k) is None else str(r.get(k)) for k, _ in cols
            ) + " |"
        )
    return "\n".join(lines)


# Every axis the sweep varies, with the value the baseline config uses.
AXES = {
    "runtime": "llamacpp", "ctx_k": 32, "kv": "f16", "fa": "on",
    "spec": "none", "n_max": None, "backend_sampling": None,
    "batch": None, "ubatch": None, "think": True, "effort": None,
    "chat_template": None, "tag": None,
}


def pick(rows, **criteria):
    """The single row matching `criteria`, unmentioned axes held at baseline.

    Reports that match on only the axes they care about will happily pick up a
    control run or a neighbouring stage's config -- the reverse-order and
    backend-sampling rows differ from the originals in fields most callers
    never name. Pinning the rest makes the lookup unambiguous, and raises
    rather than guessing when it is not.
    """
    unknown = set(criteria) - set(AXES)
    if unknown:
        raise KeyError(f"pick() got non-axis criteria: {sorted(unknown)}")
    want = dict(AXES)
    want.update(criteria)
    hits = [r for r in rows if all(r.get(k) == v for k, v in want.items())]
    if len(hits) > 1:
        raise AssertionError(
            f"pick({criteria}) is ambiguous: {[h['id'] for h in hits]}"
        )
    return hits[0] if hits else None


def main():
    data = load()
    rs = rows(data)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        rs = [r for r in rs if any(a in r["id"] for a in args)]

    meta = data.get("_meta", {})
    print(f"\n{meta.get('model', '')}  |  {meta.get('gpu', '')}  |  "
          f"llama.cpp {meta.get('llama_build', '')}")
    bases = [r["baseline_mib"] for r in rs if r.get("baseline_mib")]
    if bases:
        print(f"idle VRAM baseline re-measured per config: "
              f"{min(bases):.0f}-{max(bases):.0f} MiB observed "
              f"(subtracted from each config's own figures)")
    print(f"{meta.get('repeats')} repeats, median reported\n")
    print_table(rs)
    print(f"\n{len(rs)} configs")

    if "--md" in sys.argv:
        with open(SUMMARY_MD, "w", encoding="utf-8") as f:
            f.write("# Qwen3.8-27B settings sweep - RTX 5090 32GB\n\n")
            f.write(f"- Model: {meta.get('model')}\n")
            f.write(f"- GPU: {meta.get('gpu')}\n")
            f.write(f"- llama.cpp build: {meta.get('llama_build')}\n")
            f.write(f"- Repeats: {meta.get('repeats')} (median reported)\n")
            f.write(f"- Idle VRAM baseline: {meta.get('baseline_vram_mib')} MiB "
                    "(subtracted from all VRAM figures)\n\n")
            f.write(to_markdown(rs) + "\n")
        print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()

"""Turn the results jsonl files into the tables the write-up needs.

Every table is printed as markdown so it can be pasted into the post without
retyping numbers, which is the step where transcription errors happen.
"""

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from models import ALL, REGISTRY

RESULTS = Path(__file__).resolve().parent.parent / "results"

LABEL = {k: REGISTRY[k]["label"] for k in REGISTRY}
ORDER = ALL


def load(name):
    p = RESULTS / name
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def by_model(rows, key=lambda r: r["model"]):
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(r)
    return d


def order_keys(d):
    return [k for k in ORDER if k in d] + [k for k in d if k not in ORDER]


# --------------------------------------------------------------------------

def report_speed():
    rows = load("speed_fwd.jsonl") + load("speed_rev.jsonl")
    decode = [r for r in rows if r.get("content")]
    if not decode:
        return
    kinds = sorted({r["content"] for r in decode})
    d = by_model(decode)
    table = []
    for k in order_keys(d):
        cells = []
        for kind in kinds:
            vals = [r["decode_tps_server"] for r in d[k]
                    if r["content"] == kind and r.get("decode_tps_server")]
            cells.append(f"{st.median(vals):.1f}" if vals else "-")
        allv = [r["decode_tps_server"] for r in d[k] if r.get("decode_tps_server")]
        spread = (max(allv) - min(allv)) / st.median(allv) * 100 if allv else 0
        table.append([LABEL.get(k, k)] + cells + [f"{spread:.1f}%"])
    print("\n## Decode tok/s by content type\n")
    print(md_table(["model"] + kinds + ["spread"], table))

    fwd = [r for r in load("speed_fwd.jsonl") if r.get("content")]
    rev = [r for r in load("speed_rev.jsonl") if r.get("content")]
    if fwd and rev:
        print("\n## Order control (forward vs reverse ladder)\n")
        df, dr = by_model(fwd), by_model(rev)
        t = []
        for k in order_keys(df):
            a = st.median([r["decode_tps_server"] for r in df[k]])
            b = st.median([r["decode_tps_server"] for r in dr.get(k, [])] or [0])
            t.append([LABEL.get(k, k), f"{a:.1f}", f"{b:.1f}",
                      f"{(b - a) / a * 100:+.1f}%" if a else "-"])
        print(md_table(["model", "forward", "reverse", "delta"], t))


def report_quality():
    rows = load("quality_hard_rescored.jsonl") or load("quality_hard.jsonl")
    if not rows:
        return
    d = by_model(rows)
    t = []
    for k in order_keys(d):
        rs = d[k]
        math = [r for r in rs if r["dept"] == "math"]
        trap = [r for r in rs if r["dept"] == "trap"]
        fmt = [r for r in rs if r["dept"] == "format"]
        trunc = sum(1 for r in rs if r.get("truncated"))
        toks = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
        t.append([
            LABEL.get(k, k),
            f"{sum(r['correct'] for r in math)}/{len(math)}",
            f"{sum(r['correct'] for r in trap)}/{len(trap)}",
            f"{sum(r['correct'] for r in fmt)}/{len(fmt)}",
            trunc,
            f"{st.median(toks):.0f}" if toks else "-",
        ])
    print("\n## Quality: math / traps caught / format\n")
    print(md_table(["model", "math", "traps", "format", "truncated",
                    "median tokens"], t))

    print("\n## Per item (math), correct fraction across seeds\n")
    ids = sorted({r["id"] for r in rows if r["dept"] == "math"},
                 key=lambda x: int(x[1:]))
    t = []
    for k in order_keys(d):
        cells = []
        for i in ids:
            sub = [r for r in d[k] if r["id"] == i]
            cells.append(f"{sum(r['correct'] for r in sub)}/{len(sub)}" if sub else "-")
        t.append([LABEL.get(k, k)] + cells)
    print(md_table(["model"] + ids, t))


def report_code():
    rows = load("code.jsonl")
    if not rows:
        return
    d = by_model(rows)
    t = []
    for k in order_keys(d):
        rs = d[k]
        solved = sum(r.get("correct", False) for r in rs)
        trunc = sum(1 for r in rs if r.get("truncated"))
        frac = [r.get("assert_frac", 0) for r in rs]
        toks = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
        t.append([LABEL.get(k, k), f"{solved}/{len(rs)}",
                  f"{st.mean(frac) * 100:.0f}%", trunc,
                  f"{st.median(toks):.0f}" if toks else "-"])
    print("\n## Code: tasks fully passed, mean assert pass rate\n")
    print(md_table(["model", "solved", "asserts", "truncated",
                    "median tokens"], t))


def report_agentic():
    rows = load("agentic_hard.jsonl")
    if not rows:
        return
    d = by_model(rows)
    t = []
    for k in order_keys(d):
        rs = d[k]
        solv = [r for r in rs if r.get("kind") != "trap"]
        trap = [r for r in rs if r.get("kind") == "trap"]
        rec = [r for r in rs if r.get("hit_reject")]
        t.append([
            LABEL.get(k, k),
            f"{sum(r.get('correct', False) for r in solv)}/{len(solv)}",
            f"{sum(r.get('correct', False) for r in trap)}/{len(trap)}",
            sum(r.get("tool_calls", 0) for r in rs),
            sum(r.get("malformed_calls", 0) for r in rs),
            f"{sum(r.get('recovered_from_reject', False) for r in rec)}/{len(rec)}" if rec else "-",
            sum(1 for r in rs if r.get("exhausted_turns")),
        ])
    print("\n## Agentic: solvable / traps / calls / malformed / recovery\n")
    print(md_table(["model", "solvable", "traps", "tool calls", "malformed",
                    "recovered", "turn-exhausted"], t))


def report_multihop():
    rows = load("multihop.jsonl")
    if not rows:
        return
    fills = sorted({r["fill_target"] for r in rows})
    d = by_model(rows)
    t = []
    for k in order_keys(d):
        cells = []
        for f in fills:
            sub = [r for r in d[k] if r["fill_target"] == f]
            cells.append(f"{sum(r['correct'] for r in sub)}/{len(sub)}" if sub else "-")
        t.append([LABEL.get(k, k)] + cells)
    print("\n## Three-hop retrieval accuracy by fill length\n")
    print(md_table(["model"] + [f"{f//1024}k" for f in fills], t))


def report_ctx():
    rows = load("usable_ctx.jsonl")
    if not rows:
        return
    d = by_model(rows)
    t = []
    for k in order_keys(d):
        rs = sorted(d[k], key=lambda r: r.get("ctx", 0))
        if not rs:
            continue
        base = rs[0].get("decode_tps") or 0
        ok = [r for r in rs if base and (r.get("decode_tps") or 0) >= 0.9 * base]
        best = max((r["ctx"] for r in ok), default=None)
        t.append([LABEL.get(k, k), f"{base:.1f}", best,
                  f"{best / 1024:.0f}k" if best else "-"])
    print("\n## Usable context (decode within 90% of small-context rate)\n")
    print(md_table(["model", "base tok/s", "ceiling (tokens)", "ceiling"], t))


def report_effort():
    rows = load("effort.jsonl")
    if not rows:
        return
    d = defaultdict(list)
    for r in rows:
        d[(r["model"], r.get("effort"))].append(r)
    t = []
    for (k, eff), rs in sorted(d.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        toks = [r["completion_tokens"] for r in rs if r.get("completion_tokens")]
        think = [r["reasoning_chars"] for r in rs if r.get("reasoning_chars")]
        t.append([LABEL.get(k, k), eff or "default",
                  f"{sum(r['correct'] for r in rs)}/{len(rs)}",
                  f"{st.median(toks):.0f}" if toks else "-",
                  f"{st.median(think):.0f}" if think else "-"])
    print("\n## Granite reasoning-effort dial\n")
    print(md_table(["model", "effort", "correct", "median tokens",
                    "median thinking chars"], t))



def report_budget():
    """Solved-within-N-output-tokens, derived from one run rather than three.

    A model that eventually gets there after 30k tokens of thinking is not
    equivalent to one that gets there in 2k, and a single pass/fail column
    hides the difference entirely.
    """
    rows = load("quality_hard_rescored.jsonl") or load("quality_hard.jsonl")
    rows = [r for r in rows if r.get("dept") in ("math", "trap")]
    if not rows:
        return
    budgets = [2048, 4096, 8192, 16384, 32768]
    d = by_model(rows)
    t = []
    for k in order_keys(d):
        rs = d[k]
        cells = []
        for b in budgets:
            n = sum(1 for r in rs if r.get("correct")
                    and (r.get("completion_tokens") or 10 ** 9) <= b)
            cells.append(f"{n}/{len(rs)}")
        secs = [r["wall_s"] for r in rs if r.get("correct") and r.get("wall_s")]
        t.append([LABEL.get(k, k)] + cells +
                 [f"{st.median(secs):.0f}s" if secs else "-",
                  f"{sum(secs) / 60:.0f} min" if secs else "-"])
    print("\n## Solved within an output-token budget (math + traps)\n")
    print(md_table(["model"] + [f"<={b // 1024}k" for b in budgets] +
                   ["median s to answer", "total time on solved"], t))

REPORTS = {"speed": report_speed, "quality": report_quality,
           "budget": report_budget,
           "code": report_code, "agentic": report_agentic,
           "multihop": report_multihop, "ctx": report_ctx,
           "effort": report_effort}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    for name, fn in REPORTS.items():
        if args.only and name not in args.only:
            continue
        fn()


if __name__ == "__main__":
    main()

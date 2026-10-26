"""Scoreboard from results/hard.jsonl, plus the audits that stop bad scores.

Two audits run before any total is printed, because both are known ways this
harness manufactures a wrong number:

  truncation  a cell that hit max_tokens is not a wrong answer, it is an
              unfinished one. Those are listed separately and excluded from
              the "clean" total rather than silently counted as failures.
  strict      graders that substring-match can mark a correct answer wrong.
              Any fail whose raw output still contains the expected value is
              flagged for a human to re-read.

Nothing here re-runs the model. Every judgement comes from the stored raw
completion, which is why they are stored verbatim.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "hard.jsonl")

# expected substrings for the strict graders, used only to flag suspect fails
SUSPECT = {
    "aime_digit_sum": ["342"],
    "modular_arithmetic": ["87"],
    "expected_value": ["42"],
    "combinatorics": ["233"],
    "logic_grid": ["Meera", "Kolkata"],
    "code_output_prediction": ["[2, 2, 2]", "[1, 2] [1, 2]"],
    "multihop_needle_64k": ["649"],
    "h2_lcm_triples": ["70"],
    "h2_subsets_nonconsecutive": ["1456"],
    "h2_digits_distinct_sum": ["3600"],
}


def main():
    if not os.path.exists(OUT):
        print("no results yet at", OUT)
        return
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    configs = []
    tasks = []
    for r in rows:
        if r["config"] not in configs:
            configs.append(r["config"])
        if r["task"] not in tasks:
            tasks.append(r["task"])

    cell = {(r["config"], r["task"], r["seed"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})

    print("=" * 78)
    print("SCOREBOARD  (%d rows, seeds %s)" % (len(rows), seeds))
    print("=" * 78)
    print("%-26s %s" % ("task", " ".join("%-16s" % c for c in configs)))
    tot = {c: 0 for c in configs}
    n = {c: 0 for c in configs}
    for t in tasks:
        line = "%-26s" % t
        for c in configs:
            marks = []
            for sd in seeds:
                r = cell.get((c, t, sd))
                if r is None:
                    marks.append("-")
                    continue
                n[c] += 1
                if r["passed"]:
                    tot[c] += 1
                    marks.append("PASS")
                elif r.get("truncated"):
                    marks.append("TRUNC")
                else:
                    marks.append("fail")
            line += " %-16s" % ",".join(marks)
        print(line)
    print("%-26s %s" % ("TOTAL", " ".join(
        "%-16s" % ("%d/%d" % (tot[c], n[c])) for c in configs)))

    print()
    print("-" * 78)
    print("AUDIT 1: truncated cells (unfinished, not wrong)")
    print("-" * 78)
    tr = [r for r in rows if r.get("truncated")]
    if not tr:
        print("  none")
    for r in tr:
        u = r.get("usage") or {}
        print("  %-14s %-26s s%d  completion_tokens=%s cap=%s"
              % (r["config"], r["task"], r["seed"],
                 u.get("completion_tokens"), (r.get("cfg") or {}).get("max_tokens")))

    print()
    print("-" * 78)
    print("AUDIT 2: fails whose raw output still contains the expected value")
    print("-" * 78)
    flagged = 0
    for r in rows:
        if r["passed"] or r["task"] not in SUSPECT:
            continue
        blob = (r.get("content") or "") + (r.get("reasoning") or "")
        hits = [s for s in SUSPECT[r["task"]] if s in blob]
        if hits:
            flagged += 1
            print("  %-14s %-26s s%d  contains %s  <- re-read before scoring"
                  % (r["config"], r["task"], r["seed"], hits))
    if not flagged:
        print("  none")

    print()
    print("-" * 78)
    print("COST AND SPEED")
    print("-" * 78)
    # Truncated cells are excluded: they report the cap, not the model's cost.
    # The game is excluded and shown separately: at ~9k tokens it swamps the
    # reasoning average and answers a different question.
    print("%-16s %6s %12s %10s %12s %10s"
          % ("config", "cells", "tok/answer*", "game tok", "median t/s", "total min"))
    for c in configs:
        rs = [r for r in rows if r["config"] == c]
        clean = [r for r in rs
                 if not r.get("truncated") and r["task"] != "threejs_game"]
        toks = [(r.get("usage") or {}).get("completion_tokens") or 0 for r in clean]
        game = next(((r.get("usage") or {}).get("completion_tokens")
                     for r in rs if r["task"] == "threejs_game"), None)
        # gen_tps is completion_tokens/total_seconds, so a long-prompt task
        # reports a rate dominated by prefill, not decode. The needle prompt is
        # ~39k tokens answered in 86, which reports 0.34 t/s and is meaningless
        # as a generation rate. Excluded from the median.
        tps = sorted(r["gen_tps"] for r in rs
                     if r["gen_tps"] and r["task"] != "multihop_needle_64k")
        secs = sum(r["seconds"] for r in rs)
        med = tps[len(tps) // 2] if tps else 0
        print("%-16s %6d %12.0f %10s %12.1f %10.1f"
              % (c, len(clean), sum(toks) / max(len(toks), 1),
                 game if game is not None else "-", med, secs / 60))
    print("  * mean completion tokens over untruncated, non-game cells only")

    sk = os.path.join(ROOT, "results", "skipped.jsonl")
    if os.path.exists(sk) and os.path.getsize(sk):
        print()
        print("-" * 78)
        print("DROPPED CELLS (never silently omitted)")
        print("-" * 78)
        for l in open(sk, encoding="utf-8"):
            r = json.loads(l)
            print("  %-14s %-26s s%s  %s"
                  % (r["config"], r["task"], r["seed"], r["reason"]))


if __name__ == "__main__":
    main()

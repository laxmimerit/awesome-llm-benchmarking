"""Aggregate results/quality.jsonl.

Scores are RECOMPUTED here from the stored raw completion using the current
scorer in quality.py, rather than trusting the score written at run time. The
raw text plus the current scorer is the single source of truth, so a scorer bug
can be fixed without spending another GPU-hour re-running the models.

Counting discipline: 12 problems x 2 seeds = 24 scored runs per model. A problem
and a scored run are different units and are never conflated below.
"""
import collections, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research"))
from quality import score as rescore  # noqa: E402

probdefs = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "research", "problems.json")))}
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "quality.jsonl"), encoding="utf-8")]

changed = 0
for r in rows:
    new, det = rescore(probdefs[r["problem"]], r["content"])
    if abs(new - r["score"]) > 1e-9:
        changed += 1
        r["rescore_note"] = "%.2f -> %.2f (%s)" % (r["score"], new, det)
    r["score"], r["detail"] = new, det

models = sorted({r["model"] for r in rows}, key=lambda m: ("27b" in m, m))
probs = sorted({r["problem"] for r in rows})
cats = sorted({r["cat"] for r in rows})
seeds = sorted({r["seed"] for r in rows})

print("problems=%d  seeds=%d  models=%d  scored runs=%d  (rescored differently: %d)"
      % (len(probs), len(seeds), len(models), len(rows), changed))
exp = len(probs) * len(seeds) * len(models)
if len(rows) != exp:
    have = {(r["model"], r["problem"], r["seed"]) for r in rows}
    missing = [(m, p, s) for m in models for p in probs for s in seeds if (m, p, s) not in have]
    print("!! MISSING %d of %d cells: %s" % (len(missing), exp, missing))
if changed:
    print("\n--- rescored cells ---")
    for r in rows:
        if "rescore_note" in r:
            print("  %-20s %-7s s%d  %s" % (r["model"], r["problem"], r["seed"], r["rescore_note"]))

by = collections.defaultdict(list)
byc = collections.defaultdict(list)
trunc = collections.Counter()
toks = collections.defaultdict(list)
for r in rows:
    by[(r["model"], r["problem"])].append(r["score"])
    byc[(r["model"], r["cat"])].append(r["score"])
    by[(r["model"], "ALL")].append(r["score"])
    if not r.get("truncated"):
        by[(r["model"], "ALL_untrunc")].append(r["score"])
    else:
        trunc[r["model"]] += 1
    u = r.get("usage") or {}
    if u.get("completion_tokens"):
        toks[r["model"]].append(u["completion_tokens"])


def avg(x):
    return sum(x) / len(x) if x else float("nan")


w = max(len(m) for m in models) + 2
print("\n=== overall ===")
print("%-*s%8s%8s%9s%8s%11s%11s" % (w, "model", "runs", "mean", "no-trunc", "trunc", "med_tok", "max_tok"))
for m in models:
    t = sorted(toks[m])
    med = t[len(t) // 2] if t else 0
    print("%-*s%8d%8.3f%9.3f%8d%11d%11d"
          % (w, m, len(by[(m, "ALL")]), avg(by[(m, "ALL")]), avg(by[(m, "ALL_untrunc")]),
             trunc[m], med, max(t) if t else 0))

print("\n=== by category ===")
print("%-*s%s" % (w, "model", "".join("%11s" % c for c in cats)))
for m in models:
    print("%-*s%s" % (w, m, "".join("%11.3f" % avg(byc[(m, c)]) for c in cats)))

print("\n=== by problem (mean over seeds) ===")
print("%-*s%s" % (w, "model", "".join("%8s" % p for p in probs)))
for m in models:
    print("%-*s%s" % (w, m, "".join("%8.2f" % avg(by[(m, p)]) for p in probs)))

print("\n=== which problems discriminate ===")
for p in probs:
    vals = [avg(by[(m, p)]) for m in models if by[(m, p)]]
    print("  %-7s min=%.2f max=%.2f spread=%.2f" % (p, min(vals), max(vals), max(vals) - min(vals)))

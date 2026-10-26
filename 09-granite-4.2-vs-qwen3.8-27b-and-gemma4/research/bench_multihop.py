"""Long-context eval: three-hop retrieval inside a padded corpus.

A single-needle test only measures whether the model can find a sentence.  This
one plants three sentences that must be chained (project -> division ->
building -> drawer), places them at roughly 10%, 50% and 90% depth, and buries
them in a corpus of structurally identical distractor records.  Guessing is not
possible: every name is invented and every answer is a two-digit number that
appears many times in the corpus attached to other buildings.

Fill lengths are measured with the server's own tokenizer for the model under
test, not estimated, because six models with four different tokenizers do not
agree on what "32k tokens" means.
"""

import argparse
import json
import random
import time
from pathlib import Path

import server
from models import REGISTRY

RESULTS = Path(__file__).resolve().parent.parent / "results"

NL = chr(10)

PROJECTS = ["Kestrel", "Vireo", "Marlin", "Aurochs", "Pellucid", "Tanager",
            "Quillon", "Bramble", "Sundew", "Yarrow", "Cassini", "Dunlin"]
DIVISIONS = ["Argus", "Belfry", "Calder", "Dovetail", "Ember", "Fathom",
             "Girdle", "Harrow", "Ingot", "Juniper", "Keel", "Lantern"]

FILLER_SENTENCES = [
    "The quarterly facilities review noted no change to the occupancy plan.",
    "Badge readers on the north stair were serviced during the maintenance window.",
    "Catering deliveries are routed through the loading bay before ten in the morning.",
    "The archived floor plans remain available from the records office on request.",
    "Recycling collection was moved to Thursday for the remainder of the quarter.",
    "Visitor parking allocations are reviewed at the start of every second month.",
    "The stationery cupboard inventory was reconciled without discrepancy.",
    "Meeting room bookings longer than four hours require facilities approval.",
    "Lift inspection certificates were renewed ahead of the statutory deadline.",
    "The internal shuttle timetable was reprinted after the route adjustment.",
]


def build_corpus(rng, n_records):
    """Distractor records with exactly the same shape as the planted facts."""
    lines = []
    for i in range(n_records):
        p = f"{rng.choice(PROJECTS)}-{rng.randint(100, 999)}"
        d = f"{rng.choice(DIVISIONS)}-{rng.randint(10, 99)}"
        b = rng.randint(10, 99)
        lines.append(f"Project {p} is run by the {d} division.")
        lines.append(f"The {d} division occupies Building {b}.")
        lines.append(f"Building {b} keeps its spare access card in drawer "
                     f"{rng.randint(10, 99)}.")
        lines.append(rng.choice(FILLER_SENTENCES))
    return lines


def make_case(rng, idx):
    """One three-hop question with its planted sentences and answer."""
    p = f"{PROJECTS[idx % len(PROJECTS)]}-{2000 + idx}"
    d = f"{DIVISIONS[idx % len(DIVISIONS)]}-{700 + idx}"
    b = 1000 + idx
    drawer = 40 + idx
    facts = [
        f"Project {p} is run by the {d} division.",
        f"The {d} division occupies Building {b}.",
        f"Building {b} keeps its spare access card in drawer {drawer}.",
    ]
    q = (f"Which drawer holds the spare access card for the building occupied "
         f"by the division that runs Project {p}?")
    return {"facts": facts, "question": q, "answer": str(drawer)}


def assemble(case, target_tokens, tokenize, rng):
    """Pad with distractors until the document hits the target token count.

    Sized from a single tokenizer measurement rather than a binary search: at a
    96k fill the search sent a dozen 400 KB bodies per case, which dominated
    the run and triggered connection resets.  One measurement establishes this
    model's tokens-per-line rate, the length follows by arithmetic, and the
    real prompt length is recorded from the server's own `prompt_tokens`, so
    nothing downstream depends on the estimate being exact.
    """
    sample = build_corpus(rng, 200)
    per_line = len(tokenize(NL.join(sample))) / len(sample)
    need = int(target_tokens / per_line)

    lines = list(sample)
    while len(lines) < need:
        lines.extend(build_corpus(rng, max(50, need - len(lines))))
    lines = lines[:need]

    for frac, fact in zip((0.10, 0.50, 0.90), case["facts"]):
        lines.insert(min(len(lines), int(len(lines) * frac)), fact)
    return NL.join(lines)


PROMPT = (
    "Below is an extract from a facilities register. Read it and answer the "
    "question at the end.\n\n"
    "--- BEGIN REGISTER ---\n{doc}\n--- END REGISTER ---\n\n"
    "Question: {q}\n\n"
    "Answer with the drawer number only, on a final line in exactly this "
    "format:\nANSWER: <number>"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--fills", type=int, nargs="*", default=[8192, 32768, 98304])
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[11])
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default="multihop.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out
    import re

    for key in args.models:
        m = REGISTRY[key]
        sampling = dict(m["sampling"])
        rows = []
        # One server per fill length.  Allocating a single context large enough
        # for the longest fill would make the short-fill runs of a model that
        # spills past VRAM execute under an already degraded allocation, which
        # would read as a small-context result.  Granite 4.2 30B spills above
        # roughly 58k, so this matters for exactly the comparison being made.
        for fill in args.fills:
            ctx = fill + 8192
            print(f"\n=== {key} multihop fill={fill} ctx={ctx} ===", flush=True)
            srv = server.Server(key, ctx=ctx, tag=f"{key}_multihop_{fill}")
            srv.start(timeout=900)
            try:
                for c in range(args.cases):
                    rng = random.Random(1000 + c)
                    case = make_case(rng, c)
                    doc = assemble(case, fill, server.tokenize, rng)
                    prompt = PROMPT.format(doc=doc, q=case["question"])
                    real_tokens = len(server.tokenize(prompt))
                    for seed in args.seeds:
                        try:
                            out = server.chat(
                                [{"role": "user", "content": prompt}],
                                n_predict=args.max_tokens,
                                sampling=dict(sampling), seed=seed)
                            ch = out["choices"][0]
                            content = ch["message"].get("content") or ""
                            if "</think>" in content:
                                content = content.split("</think>")[-1]
                            found = re.findall(r"ANSWER:\s*(\d+)", content)
                            got = found[-1] if found else None
                            usage = out.get("usage", {})
                            r = {
                                "correct": got == case["answer"],
                                "got": got, "expected": case["answer"],
                                "prompt_tokens": usage.get("prompt_tokens"),
                                "completion_tokens": usage.get("completion_tokens"),
                                "prefill_tps": (out.get("timings") or {}).get("prompt_per_second"),
                                "truncated": ch.get("finish_reason") == "length",
                                "wall_s": round(out["_wall_s"], 2),
                            }
                        except Exception as e:
                            r = {"correct": False, "error": str(e)[:300],
                                 "wall_s": 0}
                        r.update({"model": key, "fill_target": fill,
                                  "fill_measured": real_tokens, "case": c,
                                  "seed": seed, "ctx": ctx})
                        rows.append(r)
                        print(f"  {'OK ' if r['correct'] else 'XX '}"
                              f"fill={fill:>6} case={c} got={r.get('got')} "
                              f"want={case['answer']} "
                              f"{r.get('wall_s', 0):7.1f}s", flush=True)
            finally:
                srv.stop()
            with open(out_path, "a", encoding="utf-8") as f:
                for r in rows[-args.cases * len(args.seeds):]:
                    f.write(json.dumps(r, default=str) + "\n")
        for fill in args.fills:
            sub = [r for r in rows if r["fill_target"] == fill]
            if sub:
                print(f"  {key} @{fill}: "
                      f"{sum(r['correct'] for r in sub)}/{len(sub)}  "
                      f"median {sorted(r['wall_s'] for r in sub)[len(sub)//2]:.1f}s",
                      flush=True)



if __name__ == "__main__":
    main()

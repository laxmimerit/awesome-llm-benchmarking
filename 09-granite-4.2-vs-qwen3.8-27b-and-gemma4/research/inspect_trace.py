"""Dump and characterise one reasoning trace.

A run that hits the token cap looks identical in the results file whether the
model was genuinely working the problem or stuck in a repetition loop, and the
two mean opposite things.  This prints the head and tail of the trace and
measures how much of it is repeated text, so "it ruminates" is a claim backed
by the trace rather than by the token count.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import server
from models import REGISTRY
from problems_hard import ALL_ITEMS, ANSWER_INSTR, TRAP_INSTR

RESULTS = Path(__file__).resolve().parent.parent / "results"


def repetition_stats(text):
    """How much of the trace is literally repeated?"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    line_counts = Counter(lines)
    dup_lines = sum(c - 1 for c in line_counts.values() if c > 1)

    words = text.split()
    grams = Counter(tuple(words[i:i + 12]) for i in range(len(words) - 12))
    dup_grams = sum(c - 1 for c in grams.values() if c > 1)

    return {
        "chars": len(text),
        "lines": len(lines),
        "unique_lines": len(line_counts),
        "duplicate_lines": dup_lines,
        "duplicate_line_frac": round(dup_lines / max(1, len(lines)), 3),
        "words": len(words),
        "duplicate_12gram_frac": round(dup_grams / max(1, len(grams)), 3),
        "most_common_line": line_counts.most_common(1)[0] if line_counts else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--id", default="M1")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    item = next(i for i in ALL_ITEMS if i["id"] == args.id)
    prompt = item["prompt"]
    if item["dept"] in ("math", "trap"):
        prompt = prompt + ANSWER_INSTR + TRAP_INSTR

    kw = {}
    if args.effort:
        kw["chat_template_kwargs"] = {"reasoning_effort": args.effort}

    srv = server.Server(args.model, ctx=args.ctx, tag=f"{args.model}_trace")
    srv.start()
    try:
        out = server.chat([{"role": "user", "content": prompt}],
                          n_predict=args.max_tokens,
                          sampling=dict(REGISTRY[args.model]["sampling"]),
                          seed=args.seed, **kw)
    finally:
        srv.stop()

    ch = out["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not reasoning and "</think>" in content:
        head, _, tail = content.partition("</think>")
        reasoning, content = head.replace("<think>", ""), tail

    print(f"model={args.model} item={args.id} effort={args.effort}")
    print(f"finish_reason={ch.get('finish_reason')} "
          f"completion_tokens={out.get('usage', {}).get('completion_tokens')}")
    print(f"reasoning chars={len(reasoning)} visible chars={len(content)}")
    print("\n--- repetition stats ---")
    print(json.dumps(repetition_stats(reasoning), indent=2, default=str))
    print("\n--- first 900 chars of reasoning ---")
    print(reasoning[:900])
    print("\n--- last 1800 chars of reasoning ---")
    print(reasoning[-1800:])
    print("\n--- visible content ---")
    print(content[:600] or "(none)")

    if args.save:
        p = RESULTS / args.save
        p.write_text(json.dumps({
            "model": args.model, "id": args.id, "effort": args.effort,
            "finish_reason": ch.get("finish_reason"),
            "usage": out.get("usage"),
            "reasoning": reasoning, "content": content,
            "stats": repetition_stats(reasoning),
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nsaved {p}")


if __name__ == "__main__":
    main()

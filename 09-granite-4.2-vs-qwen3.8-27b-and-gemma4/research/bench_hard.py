"""Quality run over the harder problem set.

Three departments in one run, deliberately interleaved:

  math   - has a correct answer, checked against a brute-force key
  trap   - has no correct answer; the only correct reply is IMPOSSIBLE
  format - graded by a predicate over the reply text, no answer key

Interleaving matters.  Traps are scored in the same pass as solvable items, so
a model that hedges everything into IMPOSSIBLE loses the math points it would
otherwise win.  Two separate rates are reported: accuracy on solvable items and
the false-answer rate on traps.

Grading never uses a judge model.
"""

import argparse
import json
import time
from pathlib import Path

import server
from models import REGISTRY
from problems_hard import (ALL_ITEMS, ANSWER_INSTR, TRAP_INSTR, extract_answer)

RESULTS = Path(__file__).resolve().parent.parent / "results"


def split_thinking(msg):
    """Return (reasoning, content) however the model emitted them.

    With --jinja llama.cpp usually fills reasoning_content.  Some templates
    still land the block inline, and a harness that only reads
    reasoning_content silently reports zero thinking for a model that is
    reasoning normally.
    """
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not reasoning and "</think>" in content:
        head, _, tail = content.partition("</think>")
        reasoning = head.replace("<think>", "")
        content = tail
    return reasoning, content


def run_item(item, sampling, max_tokens, seed, effort=None, no_think=False):
    prompt = item["prompt"]
    if item["dept"] in ("math", "trap"):
        prompt = prompt + ANSWER_INSTR + TRAP_INSTR

    tmpl = {}
    if effort is not None:
        # llama.cpp passes these straight into the jinja template render, which
        # is the only honest way to test a model's own effort dial.
        tmpl["reasoning_effort"] = effort
    if no_think:
        tmpl["enable_thinking"] = False
    kw = {"chat_template_kwargs": tmpl} if tmpl else {}

    out = server.chat([{"role": "user", "content": prompt}],
                      n_predict=max_tokens, sampling=dict(sampling), seed=seed,
                      **kw)
    choice = out["choices"][0]
    reasoning, content = split_thinking(choice["message"])
    usage = out.get("usage", {})

    if item["dept"] == "format":
        ans = None
        correct = bool(item["checker"](content))
    else:
        ans = extract_answer(content)
        correct = bool(ans is not None and item["checker"](ans))

    return {
        "id": item["id"],
        "dept": item["dept"],
        "tier": item["tier"],
        "correct": correct,
        "answered_anyway": (item["dept"] == "trap" and ans is not None
                            and not correct),
        "answer_raw": (ans or "")[:200],
        "expected": str(item.get("answer", ""))[:200],
        "reply_head": content.strip()[:300],
        "finish_reason": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length",
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "reasoning_frac_chars": round(
            len(reasoning) / max(1, len(reasoning) + len(content)), 3),
        "wall_s": round(out["_wall_s"], 2),
        "seed": seed,
        "effort": effort,
        "no_think": no_think,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 22])
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--sampling-json", default=None,
                    help="JSON object overriding the model's card sampling")
    ap.add_argument("--effort", default=None,
                    help="value passed as reasoning_effort into the template")
    ap.add_argument("--no-think", action="store_true",
                    help="pass enable_thinking=false into the template")
    ap.add_argument("--ids", nargs="*", default=None,
                    help="restrict to specific item ids")
    ap.add_argument("--dept", nargs="*", default=None,
                    help="restrict to math / trap / format")
    ap.add_argument("--out", default="quality_hard.jsonl")
    args = ap.parse_args()

    items = ALL_ITEMS
    if args.dept:
        items = [i for i in items if i["dept"] in args.dept]
    if args.ids:
        items = [i for i in items if i["id"] in args.ids]

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        m = REGISTRY[key]
        sampling = dict(m["sampling"])
        if args.sampling_json:
            sampling = json.loads(args.sampling_json)
        tag = (f"{key}_hard" + (f"_{args.effort}" if args.effort else "")
               + ("_nothink" if args.no_think else ""))
        print(f"\n=== {key} effort={args.effort} sampling={sampling} ===",
              flush=True)
        srv = server.Server(key, ctx=args.ctx, tag=tag)
        srv.start()
        rows = []
        try:
            for seed in args.seeds:
                for item in items:
                    t0 = time.time()
                    try:
                        r = run_item(item, sampling, args.max_tokens, seed,
                                     args.effort, args.no_think)
                    except Exception as e:          # keep the sweep alive
                        r = {"id": item["id"], "dept": item["dept"],
                             "tier": item["tier"], "correct": False,
                             "error": str(e)[:300], "seed": seed,
                             "effort": args.effort,
                             "no_think": args.no_think}
                    r.update({"model": key, "sampling": sampling})
                    rows.append(r)
                    flag = "OK " if r.get("correct") else "XX "
                    print(f"  {flag}{item['dept']:6} {item['id']:4} seed={seed} "
                          f"tok={str(r.get('completion_tokens')):>6} "
                          f"think={r.get('reasoning_frac_chars', 0)} "
                          f"{time.time()-t0:6.1f}s"
                          f"{' TRUNC' if r.get('truncated') else ''}",
                          flush=True)
        finally:
            srv.stop()

        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")

        solvable = [r for r in rows if r["dept"] != "trap"]
        traps = [r for r in rows if r["dept"] == "trap"]
        print(f"  {key}: solvable {sum(r['correct'] for r in solvable)}/"
              f"{len(solvable)}, traps caught {sum(r['correct'] for r in traps)}/"
              f"{len(traps)}", flush=True)


if __name__ == "__main__":
    main()

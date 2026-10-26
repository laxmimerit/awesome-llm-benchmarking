"""Accuracy and thinking economics.

Grading is objective: every problem ends with an 'ANSWER: <x>' line checked by
a pure function, so no judge model sits between the run and the score.

What gets recorded per run:
  * correct / incorrect
  * reasoning tokens vs answer tokens (the thinking-economics split)
  * whether the run died by hitting the token cap rather than finishing, which
    is the failure mode that dominated the earlier Qwen work
"""

import argparse
import json
import time
from pathlib import Path

import server
from models import REGISTRY
from problems import ANSWER_INSTR, PROBLEMS, PROBLEMS_HARD, extract_answer

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run_problem(prob, sampling, max_tokens, seed):
    msgs = [{"role": "user", "content": prob["prompt"] + ANSWER_INSTR}]
    body_sampling = dict(sampling)
    out = server.chat(msgs, n_predict=max_tokens, sampling=body_sampling, seed=seed)
    choice = out["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = out.get("usage", {})
    ans = extract_answer(content)
    # A model whose template puts thinking inline still gets counted properly.
    if not reasoning and "</think>" in content:
        head, _, tail = content.partition("</think>")
        reasoning = head.replace("<think>", "")
        content = tail
        ans = extract_answer(content)
    return {
        "id": prob["id"],
        "dept": prob["dept"],
        "correct": bool(ans is not None and prob["checker"](ans)),
        "answer_raw": (ans or "")[:200],
        "expected": str(prob.get("answer"))[:200],
        "finish_reason": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length",
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "reasoning_frac_chars": (
            round(len(reasoning) / max(1, len(reasoning) + len(content)), 3)
        ),
        "wall_s": round(out["_wall_s"], 2),
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--tier", choices=["base", "hard", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 22])
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--sampling", choices=["default", "alt"], default="default")
    ap.add_argument("--out", default="quality.jsonl")
    args = ap.parse_args()

    # Seeds are spent where they buy something. The base tier is close to
    # saturated for capable models, so a second seed there mostly re-confirms a
    # pass; the hard tier is where runs actually diverge, so it gets both.
    tiers = []
    if args.tier in ("base", "both"):
        tiers.append(("base", PROBLEMS, args.seeds[:1]))
    if args.tier in ("hard", "both"):
        tiers.append(("hard", PROBLEMS_HARD, args.seeds))

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        m = REGISTRY[key]
        sampling = dict(m["sampling_alt"] if args.sampling == "alt" and "sampling_alt" in m
                        else m["sampling"])
        print(f"\n=== {key} sampling={args.sampling} {sampling} ===", flush=True)
        srv = server.Server(key, ctx=args.ctx, tag=f"{key}_quality")
        srv.start()
        rows = []
        try:
            for tier_name, probs, tier_seeds in tiers:
                for seed in tier_seeds:
                    for prob in probs:
                        t0 = time.time()
                        r = run_problem(prob, sampling, args.max_tokens, seed)
                        r.update({"model": key, "tier": tier_name,
                                  "sampling_profile": args.sampling,
                                  "sampling": sampling})
                        rows.append(r)
                        flag = "OK " if r["correct"] else "XX "
                        trunc = " TRUNC" if r["truncated"] else ""
                        print(f"  {flag}{tier_name:4} {prob['id']:16} seed={seed} "
                              f"tok={r['completion_tokens']:>6} "
                              f"think={r['reasoning_frac_chars']:.2f} "
                              f"{time.time()-t0:6.1f}s{trunc}", flush=True)
        finally:
            srv.stop()
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        n_ok = sum(r["correct"] for r in rows)
        print(f"  {key}: {n_ok}/{len(rows)} correct", flush=True)


if __name__ == "__main__":
    main()

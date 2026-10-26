"""Long-context retrieval: does the 262k claim survive a 32 GB card?

Two things are measured together, because separately they mislead:
  * accuracy - can the model still find a fact at this depth
  * cost     - what prefill and decode speed, and what VRAM, that depth costs

A needle is placed at a controlled fraction through the filler, so "lost in the
middle" shows up as a position effect rather than a single pass/fail.
"""

import argparse
import json
import re
import time
from pathlib import Path

import server
from models import REGISTRY
from workloads import FILLER, NEEDLES, NEEDLE_QUESTION, NEEDLE_TEMPLATE

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_document(target_tokens, needle_text, depth_frac, tok_cache):
    """Filler of ~target_tokens with the needle at depth_frac through it."""
    per_chunk = tok_cache["per_chunk"]
    n_chunks = max(2, int(target_tokens / per_chunk))
    chunks = [FILLER] * n_chunks
    pos = min(n_chunks - 1, max(0, int(n_chunks * depth_frac)))
    chunks.insert(pos, "\n\n" + needle_text + "\n\n")
    return "".join(chunks)


def measure_chunk_tokens():
    n = len(server.tokenize(FILLER))
    return {"per_chunk": n}


def run_depth(model_key, target_tokens, depth_frac, needle, sampling, max_tokens):
    place, code = needle
    needle_text = NEEDLE_TEMPLATE.format(place=place, code=code)
    tok_cache = measure_chunk_tokens()
    doc = build_document(target_tokens, needle_text, depth_frac, tok_cache)
    question = NEEDLE_QUESTION.format(place=place)
    prompt = (
        "Read the following document carefully.\n\n<document>\n"
        + doc + "\n</document>" + question
    )
    n_tok = len(server.tokenize(prompt))

    t0 = time.perf_counter()
    out = server.chat(
        [{"role": "user", "content": prompt}],
        n_predict=max_tokens, sampling=sampling, seed=17,
    )
    wall = time.perf_counter() - t0
    choice = out["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not reasoning and "</think>" in content:
        reasoning, _, content = content.partition("</think>")
    usage = out.get("usage") or {}
    found = bool(re.search(re.escape(code), content, re.IGNORECASE))

    ts = out.get("timings") or {}
    return {
        "model": model_key,
        "target_tokens": target_tokens,
        "actual_prompt_tokens": n_tok,
        "depth_frac": depth_frac,
        "needle_place": place,
        "needle_code": code,
        "found": found,
        "answer": content.strip()[:200],
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "prefill_tps": ts.get("prompt_per_second"),
        "decode_tps": ts.get("predicted_per_second"),
        "wall_s": round(wall, 2),
        "finish": choice.get("finish_reason"),
        "gpu": server.gpu_state(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--depths", type=int, nargs="*",
                    default=[4096, 16384, 32768, 65536, 131072])
    ap.add_argument("--fracs", type=float, nargs="*", default=[0.1, 0.5, 0.9])
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--ctx-headroom", type=int, default=8192)
    ap.add_argument("--out", default="longctx.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        sampling = dict(REGISTRY[key]["sampling"])
        max_depth = max(args.depths)
        ctx = max_depth + args.ctx_headroom
        print(f"\n=== {key} long context, ctx={ctx} ===", flush=True)
        srv = server.Server(key, ctx=ctx, tag=f"{key}_longctx")
        try:
            srv.start(timeout=900)
        except Exception as e:
            print(f"  cannot load at ctx={ctx}: {str(e)[:300]}", flush=True)
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": key, "ctx": ctx, "ok": False,
                                    "error": str(e)[:600]}) + "\n")
            continue
        print(f"  loaded, vram delta {srv.loaded_mb - srv.baseline_mb} MiB", flush=True)
        rows = []
        try:
            for i, depth in enumerate(args.depths):
                for j, frac in enumerate(args.fracs):
                    needle = NEEDLES[(i + j) % len(NEEDLES)]
                    try:
                        r = run_depth(key, depth, frac, needle, sampling,
                                      args.max_tokens)
                    except Exception as e:
                        r = {"model": key, "target_tokens": depth,
                             "depth_frac": frac, "found": False,
                             "error": str(e)[:400]}
                    rows.append(r)
                    mark = "OK " if r.get("found") else "XX "
                    print(f"  {mark}{depth:>7} tok @ {frac:.0%}  "
                          f"prefill {r.get('prefill_tps') or 0:8.1f} tok/s  "
                          f"decode {r.get('decode_tps') or 0:6.2f}  "
                          f"think {r.get('reasoning_chars')} ch  "
                          f"{r.get('wall_s')}s", flush=True)
        finally:
            srv.stop()
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        n_ok = sum(bool(r.get("found")) for r in rows)
        print(f"  {key}: {n_ok}/{len(rows)} needles found", flush=True)


if __name__ == "__main__":
    main()

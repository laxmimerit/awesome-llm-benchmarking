"""Decode throughput and prefill speed, measured identically for every model.

Design notes that matter for trusting the numbers:

* Decode uses the raw /completion endpoint with ignore_eos, so each model
  produces exactly n_predict tokens and no chat template or stop condition can
  shorten a run and inflate tok/s.
* Timings come from llama.cpp's server-side `timings` block AND client wall
  time. Both are recorded; they should agree within a few percent.
* Each configuration is measured forward and then again in reverse order, so a
  power-capped GPU drifting over a session cannot masquerade as a model
  difference.
"""

import argparse
import json
import time
from pathlib import Path

import server
from models import REGISTRY
from workloads import DECODE_PROMPTS, FILLER

RESULTS = Path(__file__).resolve().parent.parent / "results"


def one_decode(prompt, n_predict, sampling, seed=1234):
    out = server.completion(
        prompt,
        n_predict=n_predict,
        sampling=sampling,
        ignore_eos=True,
        seed=seed,
        cache_prompt=False,
    )
    t = out["timings"]
    return {
        "prompt_n": t["prompt_n"],
        "prompt_ms": t["prompt_ms"],
        "prompt_tps_server": t.get("prompt_per_second"),
        "predicted_n": t["predicted_n"],
        "predicted_ms": t["predicted_ms"],
        "decode_tps_server": t.get("predicted_per_second"),
        "wall_s": out["_wall_s"],
        "decode_tps_wall": t["predicted_n"] / out["_wall_s"],
        "stop_type": out.get("stop_type"),
    }


def decode_sweep(model_key, n_predict=512, reps=2, warmup=True):
    """tok/s per content type, forward then reverse order."""
    m = REGISTRY[model_key]
    sampling = dict(m["sampling"])
    rows = []

    if warmup:
        one_decode(DECODE_PROMPTS["counting"], 64, sampling)

    kinds = list(DECODE_PROMPTS)
    for rep in range(reps):
        order = kinds if rep % 2 == 0 else list(reversed(kinds))
        for kind in order:
            g = server.gpu_state()
            r = one_decode(DECODE_PROMPTS[kind], n_predict, sampling)
            r.update({
                "model": model_key, "content": kind, "rep": rep,
                "order": "forward" if rep % 2 == 0 else "reverse",
                "n_predict": n_predict, "gpu": g,
                "throttle": server.throttle_reason(),
            })
            rows.append(r)
            print(f"  {model_key:18} {kind:9} rep{rep} "
                  f"{r['decode_tps_server']:7.2f} tok/s (server) "
                  f"{r['decode_tps_wall']:7.2f} (wall) "
                  f"{g['temp_c']:.0f}C {g['power_w']:.0f}W", flush=True)
    return rows


_CHUNK_TOKENS = {}


def build_prompt_of_tokens(target_tokens, model_key):
    """Filler repeated to approximately target_tokens.

    Deliberately approximate: one tokenize call establishes the tokens-per-chunk
    rate for this model, and every depth after that is arithmetic. The exact
    length is never assumed anyway, since the server reports the real prompt_n
    and that is what every rate is computed against. Binary-searching for an
    exact token count cost ~18 tokenize round trips per depth per model and
    bought nothing.
    """
    if model_key not in _CHUNK_TOKENS:
        _CHUNK_TOKENS[model_key] = max(1, len(server.tokenize(FILLER)))
    per_chunk = _CHUNK_TOKENS[model_key]
    n = max(1, round(target_tokens / per_chunk))
    return FILLER * n


def prefill_sweep(model_key, depths, n_predict=32):
    """Prompt-processing tok/s and decode tok/s at each context depth."""
    m = REGISTRY[model_key]
    sampling = dict(m["sampling"])
    rows = []
    for depth in depths:
        prompt = build_prompt_of_tokens(depth, model_key)
        g0 = server.gpu_state()
        r = one_decode(prompt, n_predict, sampling)
        g1 = server.gpu_state()
        r.update({
            "model": model_key, "depth_target": depth, "phase": "prefill",
            "gpu_before": g0, "gpu_after": g1,
            "prefill_tps_wall": r["prompt_n"] / (r["prompt_ms"] / 1000.0),
            "throttle": server.throttle_reason(),
        })
        rows.append(r)
        print(f"  {model_key:18} depth={depth:>7} "
              f"prefill {r['prompt_tps_server']:8.1f} tok/s  "
              f"decode {r['decode_tps_server']:6.2f} tok/s  "
              f"vram {g1['mem_used_mb']:.0f} MiB", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mode", choices=["decode", "prefill", "both"], default="decode")
    ap.add_argument("--depths", type=int, nargs="*",
                    default=[512, 2048, 8192, 16384, 32768])
    ap.add_argument("--out", default="speed.jsonl")
    ap.add_argument("--flash-attn", default="on")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        ctx = args.ctx
        if args.mode in ("prefill", "both"):
            ctx = max(ctx, max(args.depths) + args.n_predict + 512)
        print(f"\n=== {key}  ctx={ctx} fa={args.flash_attn} ===", flush=True)
        srv = server.Server(key, ctx=ctx, flash_attn=args.flash_attn,
                            tag=f"{key}_speed")
        t0 = time.time()
        srv.start()
        load_s = time.time() - t0
        weights_mb = srv.loaded_mb - srv.baseline_mb
        print(f"  loaded in {load_s:.1f}s  vram delta {weights_mb} MiB "
              f"(baseline {srv.baseline_mb})", flush=True)
        meta = {"model": key, "ctx": ctx, "flash_attn": args.flash_attn,
                "load_s": load_s, "vram_baseline_mb": srv.baseline_mb,
                "vram_loaded_mb": srv.loaded_mb, "vram_delta_mb": weights_mb,
                "kind": "meta", "ts": time.time()}
        rows = [meta]
        try:
            if args.mode in ("decode", "both"):
                rows += decode_sweep(key, args.n_predict, args.reps)
            if args.mode in ("prefill", "both"):
                rows += prefill_sweep(key, args.depths)
        finally:
            srv.stop()
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  wrote {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()

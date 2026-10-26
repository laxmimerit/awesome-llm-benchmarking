"""Does merely ALLOCATING a large context slow decoding down?

The long-context phase measured Gemma decoding at 16.5 tok/s where the speed
phase had it at 68.9. The two runs differed only in the context the server was
started with, so the effect was inferred across phases rather than measured.
This isolates it: identical short prompt, identical short generation, the only
variable being -c at launch.

Reported as a ratio against each model's own small-context number, so a model
that is simply slow is not confused with a model that degrades.
"""

import argparse
import json
import time
from pathlib import Path

import server
from models import REGISTRY
from workloads import DECODE_PROMPTS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def measure(model_key, ctx, n_predict, sampling):
    srv = server.Server(model_key, ctx=ctx, tag=f"{model_key}_ctxalloc_{ctx}")
    row = {"model": model_key, "ctx": ctx, "n_predict": n_predict}
    try:
        srv.start(timeout=900)
    except Exception as e:
        row.update({"ok": False, "error": str(e)[:400]})
        return row
    try:
        # warm up, then measure a short generation on a short prompt
        server.completion(DECODE_PROMPTS["counting"], n_predict=32,
                          sampling=sampling, ignore_eos=True, seed=5,
                          cache_prompt=False)
        out = server.completion(DECODE_PROMPTS["code"], n_predict=n_predict,
                                sampling=sampling, ignore_eos=True, seed=5,
                                cache_prompt=False)
        t = out["timings"]
        g = server.gpu_state()
        row.update({
            "ok": True,
            "decode_tps": t.get("predicted_per_second"),
            "prefill_tps": t.get("prompt_per_second"),
            "prompt_n": t["prompt_n"],
            "vram_delta_mb": srv.loaded_mb - srv.baseline_mb,
            "vram_used_mb": g["mem_used_mb"],
            "vram_total_mb": g["mem_total_mb"],
            "temp_c": g["temp_c"], "power_w": g["power_w"],
            "throttle": server.throttle_reason(),
        })
    finally:
        srv.stop()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--small-ctx", type=int, default=8192)
    ap.add_argument("--large-ctx", type=int, default=131072)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--out", default="ctxalloc.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        sampling = dict(REGISTRY[key]["sampling"])
        print(f"\n=== {key} ===", flush=True)
        rows = []
        for ctx in (args.small_ctx, args.large_ctx):
            r = measure(key, ctx, args.n_predict, sampling)
            rows.append(r)
            if r.get("ok"):
                print(f"  ctx={ctx:>7}  decode {r['decode_tps']:7.2f} tok/s  "
                      f"vram {r['vram_used_mb']:.0f}/{r['vram_total_mb']:.0f} MiB  "
                      f"{r['temp_c']:.0f}C", flush=True)
            else:
                print(f"  ctx={ctx:>7}  FAILED: {r.get('error','')[:150]}", flush=True)
        ok = [r for r in rows if r.get("ok")]
        if len(ok) == 2:
            ratio = ok[1]["decode_tps"] / ok[0]["decode_tps"]
            print(f"  -> decode at large ctx is {ratio:.2f}x the small-ctx rate "
                  f"({(1-ratio)*100:+.0f}% change)", flush=True)
            rows.append({"model": key, "kind": "ratio", "ratio": round(ratio, 4),
                         "small_ctx": args.small_ctx, "large_ctx": args.large_ctx,
                         "small_tps": ok[0]["decode_tps"],
                         "large_tps": ok[1]["decode_tps"]})
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()

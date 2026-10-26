"""Concurrency scaling: aggregate throughput as parallel slots increase.

This is the axis where a 22 GB MoE can beat a 6 GB dense model, and it is
almost never published. The server is restarted for every slot count because
-np changes how the KV cache is partitioned.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import server
from models import REGISTRY
from workloads import DECODE_PROMPTS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def fire(prompt, n_predict, sampling, seed):
    out = server.completion(prompt, n_predict=n_predict, sampling=sampling,
                            ignore_eos=True, seed=seed, cache_prompt=False)
    t = out["timings"]
    return {"predicted_n": t["predicted_n"], "predicted_ms": t["predicted_ms"],
            "decode_tps_server": t.get("predicted_per_second"),
            "wall_s": out["_wall_s"]}


def run_level(model_key, n_par, n_predict, content, sampling):
    prompt = DECODE_PROMPTS[content]
    # warm one slot so the first timing is not a cold graph build
    fire(prompt, 32, sampling, 1)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        futs = [ex.submit(fire, prompt, n_predict, sampling, 1000 + i)
                for i in range(n_par)]
        res = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    total_tokens = sum(r["predicted_n"] for r in res)
    return {
        "model": model_key, "parallel": n_par, "content": content,
        "n_predict": n_predict,
        "wall_s": round(wall, 3),
        "total_tokens": total_tokens,
        "aggregate_tps": round(total_tokens / wall, 2),
        "per_stream_tps_mean": round(
            sum(r["decode_tps_server"] for r in res) / len(res), 2),
        "per_stream_tps_min": round(min(r["decode_tps_server"] for r in res), 2),
        "gpu": server.gpu_state(),
        "throttle": server.throttle_reason(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--levels", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--content", default="code")
    ap.add_argument("--ctx-per-slot", type=int, default=4096)
    ap.add_argument("--out", default="concurrency.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        sampling = dict(REGISTRY[key]["sampling"])
        for n_par in args.levels:
            ctx = args.ctx_per_slot * n_par
            print(f"\n=== {key} np={n_par} ctx={ctx} ===", flush=True)
            srv = server.Server(key, ctx=ctx, parallel=n_par,
                                tag=f"{key}_conc{n_par}")
            try:
                srv.start()
            except Exception as e:
                row = {"model": key, "parallel": n_par, "ok": False,
                       "error": str(e)[:400]}
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  FAILED: {str(e)[:200]}", flush=True)
                continue
            try:
                r = run_level(key, n_par, args.n_predict, args.content, sampling)
                r["ok"] = True
                r["vram_delta_mb"] = srv.loaded_mb - srv.baseline_mb
                print(f"  aggregate {r['aggregate_tps']:8.2f} tok/s  "
                      f"per-stream {r['per_stream_tps_mean']:6.2f}  "
                      f"vram {r['vram_delta_mb']} MiB", flush=True)
            finally:
                srv.stop()
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()

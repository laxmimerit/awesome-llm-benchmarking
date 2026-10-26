"""The largest context each model can actually USE, not merely load.

Why this replaces the load-success test:

On Windows, a CUDA allocation that exceeds VRAM does not fail. The driver backs
it with system memory over PCIe, llama.cpp reports a clean start, and the model
runs at a fraction of its speed. Measured directly on Gemma 4-31B: VRAM usage
plateaued at ~31.5 GB while the KV cache kept growing by gigabytes, and decode
fell from 69.7 tok/s to 16.5. Nothing anywhere reported an error.

So "max context" is defined here as the largest context at which the model still
decodes at >=90% of its own small-context rate. That is a performance cliff, not
an allocation failure, and it is the number a user actually cares about.
"""

import argparse
import json
from pathlib import Path

import server
from models import REGISTRY
from workloads import DECODE_PROMPTS

RESULTS = Path(__file__).resolve().parent.parent / "results"
ADVERTISED = 262144
KEEP_FRACTION = 0.90

# Each model's own trained context, read from the GGUF census.  Probing a model
# past its nominal length measures rope extrapolation, not a context ceiling,
# so the search is capped per model rather than at a shared 256k.
def nominal_ctx(model_key, default=ADVERTISED):
    try:
        arch = json.loads((RESULTS / "arch.json").read_text())
        return int(arch[model_key]["ctx"])
    except Exception:
        return default


def decode_at(model_key, ctx, sampling, n_predict=192):
    """Start at this context, generate from a SHORT prompt, report the rate."""
    srv = server.Server(model_key, ctx=ctx, tag=f"{model_key}_usable_{ctx}",
                        extra=["-lv", "4"])
    row = {"model": model_key, "ctx": ctx}
    try:
        srv.start(timeout=900)
    except Exception as e:
        row.update({"ok": False, "error": str(e)[:300]})
        return row
    try:
        server.completion(DECODE_PROMPTS["counting"], n_predict=32,
                          sampling=sampling, ignore_eos=True, seed=5,
                          cache_prompt=False)
        out = server.completion(DECODE_PROMPTS["code"], n_predict=n_predict,
                                sampling=sampling, ignore_eos=True, seed=5,
                                cache_prompt=False)
        g = server.gpu_state()
        row.update({
            "ok": True,
            "decode_tps": out["timings"].get("predicted_per_second"),
            "vram_used_mb": g["mem_used_mb"],
            "vram_total_mb": g["mem_total_mb"],
            "vram_delta_mb": srv.loaded_mb - srv.baseline_mb,
            "headroom_mb": g["mem_total_mb"] - g["mem_used_mb"],
        })
    finally:
        srv.stop()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--base-ctx", type=int, default=8192)
    ap.add_argument("--out", default="usable_ctx.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        sampling = dict(REGISTRY[key]["sampling"])
        print(f"\n=== {key} ===", flush=True)
        rows = []

        base = decode_at(key, args.base_ctx, sampling)
        rows.append(base)
        if not base.get("ok"):
            print(f"  baseline failed: {base.get('error')}", flush=True)
            continue
        ref = base["decode_tps"]
        floor = ref * KEEP_FRACTION
        print(f"  baseline @{args.base_ctx}: {ref:.1f} tok/s "
              f"(floor {floor:.1f}), vram {base['vram_used_mb']:.0f} MiB",
              flush=True)

        # Climb geometrically until the rate drops below the floor, then bisect.
        top = nominal_ctx(key)
        lo_ok, hi_bad = args.base_ctx, None
        probe = min(top, 65536)
        while len(rows) < 9:
            r = decode_at(key, probe, sampling)
            rows.append(r)
            good = r.get("ok") and r.get("decode_tps", 0) >= floor
            if r.get("ok"):
                print(f"  ctx={probe:>7}  {r['decode_tps']:7.2f} tok/s  "
                      f"vram {r['vram_used_mb']:.0f}/{r['vram_total_mb']:.0f} MiB  "
                      f"{'ok' if good else 'DEGRADED'}", flush=True)
            else:
                print(f"  ctx={probe:>7}  FAILED TO LOAD", flush=True)
            if good:
                lo_ok = probe
                if probe >= top:
                    break
                probe = min(top, int(probe * 2))
            else:
                hi_bad = probe
                break

        while (hi_bad and len(rows) < 12
               and hi_bad - lo_ok > max(4096, lo_ok * 0.08)):
            mid = ((lo_ok + hi_bad) // 2 // 1024) * 1024
            r = decode_at(key, mid, sampling)
            rows.append(r)
            good = r.get("ok") and r.get("decode_tps", 0) >= floor
            print(f"  ctx={mid:>7}  {r.get('decode_tps') or 0:7.2f} tok/s  "
                  f"{'ok' if good else 'DEGRADED'}", flush=True)
            if good:
                lo_ok = mid
            else:
                hi_bad = mid

        verdict = {
            "model": key, "kind": "usable_ctx",
            "baseline_ctx": args.base_ctx, "baseline_tps": ref,
            "usable_max_ctx": lo_ok,
            "first_degraded_ctx": hi_bad,
            "nominal_ctx": top,
            "holds_full_window": lo_ok >= top,
            "fraction_of_nominal": round(lo_ok / top, 3),
        }
        rows.append(verdict)
        print(f"  -> usable max ctx: {lo_ok:,} "
              f"({lo_ok/top*100:.0f}% of its own nominal {top:,})"
              + (f", degrades at {hi_bad:,}" if hi_bad else ""), flush=True)

        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()

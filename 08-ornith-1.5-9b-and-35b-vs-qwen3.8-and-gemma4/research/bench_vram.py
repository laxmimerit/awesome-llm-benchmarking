"""VRAM accounting: weights, KV cache, compute buffer, and the context ceiling.

The question this answers is the one the model cards never do: on a single
32 GB card, how much context does each model actually get?

Three sources are cross-checked rather than trusted individually:
  * llama.cpp's own log lines (KV self size, compute buffer, model buffer)
  * nvidia-smi delta from the pre-load baseline
  * measured KV growth between two context sizes
"""

import argparse
import json
import re
import time
from pathlib import Path

import server
from models import REGISTRY

RESULTS = Path(__file__).resolve().parent.parent / "results"

RE_MODEL_BUF = re.compile(r"load_tensors:\s+(\S+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB")
RE_KV = re.compile(r"llama_kv_cache[^:]*:\s+(\S+)\s+KV buffer size\s*=\s*([\d.]+)\s*MiB")
# One line per cache. Models with sliding-window attention (gemma4, muse) emit
# two: a full cache and a windowed one, with different cell counts.
RE_KV_SELF = re.compile(
    r"llama_kv_cache:\s*size\s*=\s*([\d.]+)\s*MiB\s*\(\s*(\d+)\s*cells,\s*"
    r"(\d+)\s*layers,\s*(\S+)\s*seqs\),\s*K\s*\(([^)]*)\):\s*([\d.]+)\s*MiB,\s*"
    r"V\s*\(([^)]*)\):\s*([\d.]+)\s*MiB"
)
RE_COMPUTE = re.compile(r"(\S+)\s+compute buffer size\s*=\s*([\d.]+)\s*MiB")
RE_CTX = re.compile(r"n_ctx\s*=\s*(\d+)")
RE_CTX_PER_SEQ = re.compile(r"n_ctx_per_seq\s*=\s*(\d+)")


def parse_log(path):
    txt = Path(path).read_text(errors="replace")
    out = {}
    model_bufs = {m.group(1): float(m.group(2)) for m in RE_MODEL_BUF.finditer(txt)}
    out["model_buffer_mib"] = model_bufs
    out["model_buffer_total_mib"] = round(sum(model_bufs.values()), 1)

    kv_bufs = {m.group(1): float(m.group(2)) for m in RE_KV.finditer(txt)}
    out["kv_buffer_mib"] = kv_bufs
    out["kv_buffer_total_mib"] = round(sum(kv_bufs.values()), 1)

    caches = []
    for m in RE_KV_SELF.finditer(txt):
        caches.append({
            "size_mib": float(m.group(1)),
            "cells": int(m.group(2)),
            "layers": int(m.group(3)),
            "seqs": m.group(4),
            "k_type": m.group(5).strip(),
            "k_mib": float(m.group(6)),
            "v_type": m.group(7).strip(),
            "v_mib": float(m.group(8)),
        })
    if caches:
        out["kv_caches"] = caches
        out["kv_self_mib"] = round(sum(c["size_mib"] for c in caches), 2)
        out["kv_layers_total"] = sum(c["layers"] for c in caches)
        out["kv_k_type"] = caches[0]["k_type"]
        out["kv_v_type"] = caches[0]["v_type"]

    comp = {m.group(1): float(m.group(2)) for m in RE_COMPUTE.finditer(txt)}
    out["compute_buffer_mib"] = comp
    out["compute_buffer_total_mib"] = round(sum(comp.values()), 1)

    m = RE_CTX.search(txt)
    if m:
        out["n_ctx_log"] = int(m.group(1))
    m = RE_CTX_PER_SEQ.search(txt)
    if m:
        out["n_ctx_per_seq_log"] = int(m.group(1))
    return out


def probe(model_key, ctx, ctk=None, ctv=None, flash_attn="on", parallel=1):
    """Load once at a given config and report every VRAM number we can get."""
    tag = f"{model_key}_vram_c{ctx}_{ctk or 'f16'}_{flash_attn}_np{parallel}"
    # -lv 4 is what makes llama.cpp print its own buffer accounting; the speed
    # runs stay at default verbosity so logging never costs throughput.
    srv = server.Server(model_key, ctx=ctx, flash_attn=flash_attn,
                        cache_type_k=ctk, cache_type_v=ctv,
                        parallel=parallel, tag=tag, extra=["-lv", "4"])
    row = {"model": model_key, "ctx": ctx, "cache_type_k": ctk or "f16",
           "cache_type_v": ctv or "f16", "flash_attn": flash_attn,
           "parallel": parallel}
    t0 = time.time()
    try:
        srv.start()
    except Exception as e:
        row["ok"] = False
        row["error"] = str(e)[:600]
        if srv.log_path and Path(srv.log_path).exists():
            row.update(parse_log(srv.log_path))
        return row
    row["ok"] = True
    row["load_s"] = round(time.time() - t0, 2)
    row["vram_baseline_mb"] = srv.baseline_mb
    row["vram_loaded_mb"] = srv.loaded_mb
    row["vram_delta_mb"] = srv.loaded_mb - srv.baseline_mb
    row.update(parse_log(srv.log_path))
    try:
        row["props_n_ctx"] = srv.props().get("default_generation_settings", {}).get("n_ctx")
    except Exception:
        pass
    srv.stop()
    return row


def kv_per_token(rows):
    """Derive bytes-per-token of KV from two loads at different ctx."""
    by_ctx = {r["ctx"]: r for r in rows if r.get("ok") and "kv_self_mib" in r}
    if len(by_ctx) < 2:
        return None
    ctxs = sorted(by_ctx)
    lo, hi = ctxs[0], ctxs[-1]
    d_mib = by_ctx[hi]["kv_self_mib"] - by_ctx[lo]["kv_self_mib"]
    d_tok = hi - lo
    return {
        "kv_bytes_per_token": round(d_mib * 1024 * 1024 / d_tok, 1),
        "kv_kib_per_1k_tokens": round(d_mib * 1024 / d_tok * 1000 / 1024, 1),
        "from_ctx": lo, "to_ctx": hi,
    }


def predict_max_ctx(model_key, headroom_mb=900):
    """Predict the context ceiling from already-measured numbers.

    Everything needed is linear and was measured in the probe phase: weights are
    fixed, KV grows at a constant bytes-per-token, and the compute buffer grows
    slowly. Binary searching for the ceiling meant ~8 model loads per model and
    landed on the same answer this arithmetic gives, so the search is replaced
    by predict-then-verify.
    """
    rows = [json.loads(l) for l in
            (RESULTS / "vram.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [r for r in rows if r.get("model") == model_key and r.get("ok")]
    kv = [r for r in rows if r.get("model") == model_key
          and r.get("kind") == "kv_per_token"]
    if not mine or not kv:
        return None
    total_mb = server.gpu_state()["mem_total_mb"]
    baseline_mb = mine[-1]["vram_baseline_mb"]
    weights_mb = mine[-1].get("model_buffer_total_mib") or mine[-1]["vram_delta_mb"]
    compute_mb = mine[-1].get("compute_buffer_total_mib") or 300
    kv_per_tok = kv[-1]["kv_bytes_per_token"]
    free_for_kv_mb = total_mb - baseline_mb - weights_mb - compute_mb - headroom_mb
    if free_for_kv_mb <= 0:
        return 0
    tokens = int(free_for_kv_mb * 1024 * 1024 / kv_per_tok)
    return {
        "predicted_max_ctx": (tokens // 1024) * 1024,
        "total_mb": total_mb, "baseline_mb": baseline_mb,
        "weights_mb": weights_mb, "compute_mb": compute_mb,
        "kv_bytes_per_token": kv_per_tok,
        "free_for_kv_mb": round(free_for_kv_mb, 1),
    }


def find_max_ctx(model_key, ctk=None, ctv=None, flash_attn="on", step_json=None):
    """Verify the predicted ceiling with at most three loads.

    Caps at the model's own advertised window: a model that can hold its full
    262k context has no meaningful "ceiling" beyond that, and probing past it
    would measure llama.cpp's RoPE handling rather than VRAM.
    """
    pred = predict_max_ctx(model_key)
    if not pred:
        return None, [], None
    advertised = 262144
    target = min(pred["predicted_max_ctx"], advertised)
    target = (target // 1024) * 1024
    tried = []
    best = None

    def attempt(ctx):
        nonlocal best
        ctx = max(1024, (ctx // 1024) * 1024)
        print(f"    trying ctx={ctx} ...", flush=True)
        r = probe(model_key, ctx, ctk, ctv, flash_attn)
        tried.append({"ctx": ctx, "ok": r.get("ok"),
                      "vram_delta_mb": r.get("vram_delta_mb"),
                      "error": (r.get("error") or "")[:200]})
        if step_json is not None:
            step_json.append(r)
        if r.get("ok") and (best is None or ctx > best):
            best = ctx
        return r.get("ok")

    # Climb until something actually fails, then bisect the gap.
    #
    # An earlier version stepped up exactly once and stopped on success, which
    # meant it reported the last PASSING probe as if it were the ceiling. Every
    # model came back with a number no probe had ever falsified, and Gemma was
    # later shown loading at nearly twice its "ceiling". A ceiling is only a
    # ceiling once the next size up has been observed to fail.
    MAX_PROBES = 8
    lo_ok, hi_fail = None, None

    if attempt(target):
        lo_ok = target
        while len(tried) < MAX_PROBES:
            if lo_ok >= advertised:
                break
            nxt = min(advertised, int(lo_ok * 1.35))
            if nxt <= lo_ok:
                break
            if attempt(nxt):
                lo_ok = nxt
            else:
                hi_fail = nxt
                break
    else:
        hi_fail = target
        step = target
        while len(tried) < MAX_PROBES:
            step = int(step * 0.75)
            if step < 4096:
                break
            if attempt(step):
                lo_ok = step
                break
            hi_fail = step

    # Narrow whatever gap is left between the largest pass and smallest fail.
    while (lo_ok and hi_fail and len(tried) < MAX_PROBES
           and hi_fail - lo_ok > max(4096, lo_ok * 0.06)):
        mid = (lo_ok + hi_fail) // 2
        if attempt(mid):
            lo_ok = mid
        else:
            hi_fail = mid

    return best, tried, {**(pred or {}), "smallest_failing_ctx": hi_fail,
                         "bounded": hi_fail is not None or best >= advertised}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--mode", choices=["probe", "maxctx", "kvquant"], default="probe")
    ap.add_argument("--ctxs", type=int, nargs="*", default=[4096, 32768])
    ap.add_argument("--out", default="vram.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        print(f"\n=== {key} :: {args.mode} ===", flush=True)
        rows = []
        if args.mode == "probe":
            for ctx in args.ctxs:
                r = probe(key, ctx)
                rows.append(r)
                if r.get("ok"):
                    print(f"  ctx={ctx:>7} weights={r.get('model_buffer_total_mib')} MiB "
                          f"kv={r.get('kv_self_mib')} MiB "
                          f"compute={r.get('compute_buffer_total_mib')} MiB "
                          f"smi_delta={r['vram_delta_mb']} MiB", flush=True)
                else:
                    print(f"  ctx={ctx:>7} FAILED: {r.get('error','')[:200]}", flush=True)
            kv = kv_per_token(rows)
            if kv:
                print(f"  -> KV {kv['kv_bytes_per_token']} bytes/token "
                      f"({kv['kv_kib_per_1k_tokens']} KiB per 1k)", flush=True)
                rows.append({"model": key, "kind": "kv_per_token", **kv})

        elif args.mode == "kvquant":
            # flash attention stays on: the f16/q8_0/q4_0 comparison is the
            # point, and an fa=off arm would triple the loads to answer a
            # question nobody deploys against.
            for ctk, ctv in [(None, None), ("q8_0", "q8_0"), ("q4_0", "q4_0")]:
                r = probe(key, args.ctxs[-1], ctk, ctv, flash_attn="on")
                rows.append(r)
                print(f"  kv={ctk or 'f16':>5} ok={r.get('ok')} "
                      f"kv_self={r.get('kv_self_mib')} MiB "
                      f"smi_delta={r.get('vram_delta_mb')} MiB", flush=True)

        elif args.mode == "maxctx":
            steps = []
            best, tried, pred = find_max_ctx(key, step_json=steps)
            if pred:
                print(f"  predicted {pred['predicted_max_ctx']} from "
                      f"weights={pred['weights_mb']:.0f} MiB, "
                      f"kv={pred['kv_bytes_per_token']/1024:.1f} KiB/tok", flush=True)
            print(f"  -> max ctx verified: {best}", flush=True)
            rows += steps
            rows.append({"model": key, "kind": "max_ctx", "max_ctx": best,
                         "predicted": pred, "tried": tried})

        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  wrote {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()

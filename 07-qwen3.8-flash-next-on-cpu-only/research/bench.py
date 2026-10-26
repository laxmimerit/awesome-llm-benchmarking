"""Run llama-bench sweeps for Qwen3.8-Flash-Next and record raw JSONL.

Every row llama-bench emits is written verbatim to results/<stage>.jsonl with a
"_run" object carrying the config we asked for, so a report can always trace a
number back to the exact command that produced it.
"""
import argparse, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_CPU  = os.path.join(ROOT, "research", "build-cpu",  "bin", "Release", "llama-bench.exe")
BIN_CUDA = os.path.join(ROOT, "research", "build-cuda", "bin", "llama-bench.exe")
MODELS = {
    "IQ1_S":   os.path.join(ROOT, "models", "UD-IQ1_S",   "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"),
    "Q2_K_XL": os.path.join(ROOT, "models", "UD-Q2_K_XL", "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"),
}
PLE = "per_layer_token_embd"


def host_facts():
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS(); m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram = {"total_gb": round(m.ullTotalPhys / 2**30, 2), "avail_gb": round(m.ullAvailPhys / 2**30, 2)}
    except Exception as e:
        ram = {"error": str(e)}
    return {"cpu": platform.processor(), "ram": ram}


def run(binary, model_key, args, stage, note, results, logs, repetitions=3):
    model = MODELS[model_key]
    if not os.path.exists(model):
        print(f"SKIP {note}: model missing {model}", flush=True)
        return
    cmd = [binary, "-m", model, "-o", "jsonl", "-r", str(repetitions), "--progress"] + args
    tag = f"{stage}|{model_key}|{note}"
    print(f"\n=== {tag} ===\n{' '.join(cmd)}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    with open(os.path.join(logs, f"bench_{stage}_{model_key}_{note}.log"), "w", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n\nSTDOUT:\n" + p.stdout + "\n\nSTDERR:\n" + p.stderr)

    meta = {"stage": stage, "quant": model_key, "note": note, "cmd": cmd,
            "wall_s": round(dt, 1), "exit": p.returncode,
            "ts": datetime.now(timezone.utc).isoformat(), "host": host_facts()}

    n = 0
    with open(os.path.join(results, f"{stage}.jsonl"), "a", encoding="utf-8") as fh:
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_run"] = meta
            fh.write(json.dumps(row) + "\n")
            n += 1
    print(f"--> exit={p.returncode} rows={n} wall={dt:.0f}s", flush=True)
    if p.returncode != 0:
        print(p.stderr[-2000:], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cpu", "gpu", "smoke", "longctx", "longctx_gpu"])
    ap.add_argument("--quant", default="IQ1_S")
    ap.add_argument("--threads", default="24")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--depths", default="0,4096")
    ap.add_argument("--ncmoe", default="48")
    a = ap.parse_args()

    results = os.path.join(ROOT, "results"); os.makedirs(results, exist_ok=True)
    logs = os.path.join(ROOT, "logs");       os.makedirs(logs, exist_ok=True)

    if a.stage == "smoke":
        run(BIN_CPU, a.quant, ["-p", "64", "-n", "16", "-t", "24"], "smoke", "loadcheck", results, logs, 1)

    elif a.stage == "cpu":
        # Pure CPU: the binary has no CUDA compiled in at all.
        run(BIN_CPU, a.quant, ["-p", "512", "-n", "128", "-t", a.threads],
            "cpu", "threadsweep", results, logs, a.reps)
        run(BIN_CPU, a.quant, ["-p", "512", "-n", "128", "-t", a.threads.split(",")[-1],
                               "-lm", "mmap,mlock"],
            "cpu", "loadmode", results, logs, a.reps)
        run(BIN_CPU, a.quant, ["-n", "128", "-d", a.depths, "-t", a.threads.split(",")[-1]],
            "cpu", "depth", results, logs, max(2, a.reps - 1))

    elif a.stage == "longctx":
        # Context scaling on pure CPU. Warmup is skipped: at 32k a warmup pass
        # alone costs ~11 min and measures nothing we are not already repeating.
        run(BIN_CPU, a.quant, ["-n", "128", "-d", a.depths, "-t", a.threads, "--no-warmup"],
            "longctx", "cpu", results, logs, a.reps)

    elif a.stage == "longctx_gpu":
        run(BIN_CUDA, a.quant, ["-n", "128", "-d", a.depths, "-t", a.threads, "--no-warmup",
                                "-ngl", "99", "-ncmoe", a.ncmoe, "-ot", f"{PLE}=CPU"],
            "longctx", f"gpu_ncmoe{a.ncmoe}", results, logs, a.reps)

    elif a.stage == "gpu":
        # Expert banks stay in RAM; everything else climbs onto the 5090.
        run(BIN_CUDA, a.quant, ["-p", "512", "-n", "128", "-t", a.threads,
                                "-ngl", "99", "-ncmoe", "48,40,32,24,16,12",
                                "-ot", f"{PLE}=CPU"],
            "gpu", "ncmoe_sweep_pleCPU", results, logs, a.reps)
        # Same ladder but letting the 26.8 GiB gather table compete for VRAM.
        run(BIN_CUDA, a.quant, ["-p", "512", "-n", "128", "-t", a.threads,
                                "-ngl", "99", "-ncmoe", "48,40"],
            "gpu", "ncmoe_sweep_pleGPU", results, logs, a.reps)
        run(BIN_CUDA, a.quant, ["-p", "512", "-n", "128", "-t", a.threads, "-ngl", "0"],
            "gpu", "ngl0_control", results, logs, a.reps)


if __name__ == "__main__":
    main()

"""Phase runner for the full sweep.

Phases are ordered so the numbers that decide everything else land first, and
so the one phase that must be paired back-to-back (speed) runs as a single
uninterrupted block across all models rather than model-by-model.

Each phase appends to its own results file and is skippable, so a crash or an
interrupt never costs more than the phase in flight.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"

ALL_MODELS = ["ornith-9b", "ornith-35b-a3b", "qwen3.8-27b", "gemma4-26b"]


def sh(args, log_name):
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / log_name
    print(f"\n$ {' '.join(str(a) for a in args)}\n   -> {log}", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"{' '.join(str(a) for a in args)} =====\n")
        f.flush()
        p = subprocess.run([sys.executable, *args], cwd=str(HERE),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace")
        f.write(p.stdout or "")
    tail = "\n".join((p.stdout or "").splitlines()[-14:])
    print(tail, flush=True)
    print(f"   [{time.time()-t0:.0f}s, exit {p.returncode}]", flush=True)
    return p.returncode


def phase_vram(models):
    # Two context sizes, not three. KV is allocated up front and grows exactly
    # linearly, so a third point only re-confirms a straight line at the cost of
    # another model load per model.
    for m in models:
        sh(["bench_vram.py", m, "--mode", "probe",
            "--ctxs", "4096", "65536", "--out", "vram.jsonl"],
           "phase_vram.log")


def phase_kvquant(models):
    for m in models:
        sh(["bench_vram.py", m, "--mode", "kvquant",
            "--ctxs", "65536", "--out", "kvquant.jsonl"], "phase_kvquant.log")


def phase_maxctx(models):
    for m in models:
        sh(["bench_vram.py", m, "--mode", "maxctx", "--out", "maxctx.jsonl"],
           "phase_maxctx.log")


def phase_speed(models):
    # Single forward pass. The reverse-order control run is deliberately not in
    # the default order: it guards against a ~4% run-order artefact, which sits
    # inside the spread the same model shows from one machine to another, so it
    # does not change any conclusion this benchmark draws. Still available as an
    # explicit phase if a result ever lands inside that margin.
    sh(["bench_speed.py", *models, "--ctx", "8192", "--n-predict", "512",
        "--reps", "1", "--mode", "decode", "--out", "speed.jsonl"],
       "phase_speed.log")


def phase_speed_reverse(models):
    sh(["bench_speed.py", *list(reversed(models)), "--ctx", "8192",
        "--n-predict", "512", "--reps", "1", "--mode", "decode",
        "--out", "speed_reverse.jsonl"], "phase_speed.log")


def phase_prefill(models):
    # Shallow depths only. 64k and 128k prefill is measured anyway inside the
    # long-context phase, on prompts that are doing useful work, so repeating
    # those depths here would be paying twice for one number.
    sh(["bench_speed.py", *models, "--n-predict", "32", "--mode", "prefill",
        "--depths", "512", "2048", "8192", "32768",
        "--out", "prefill.jsonl"], "phase_prefill.log")


def phase_mtp(models):
    sh(["check_mtp.py", *models, "--n-predict", "512", "--out", "mtp_check.json"],
       "phase_mtp.log")


def phase_quality(models):
    # bench_quality spends one seed on the base tier and both on the hard tier.
    for m in models:
        sh(["bench_quality.py", m, "--tier", "both", "--seeds", "11", "22",
            "--max-tokens", "16384", "--ctx", "32768", "--out", "quality.jsonl"],
           "phase_quality.log")


def phase_agentic(models):
    # Two seeds kept here on purpose: this is the axis the Ornith card makes its
    # loudest claim on, and single-seed agentic scores are noisy.
    for m in models:
        sh(["bench_agentic.py", m, "--seeds", "3", "5", "--out", "agentic.jsonl"],
           "phase_agentic.log")


def phase_longctx(models):
    # Three depths, two positions. The dropped depths (4k, 32k) sit on a curve
    # the remaining points already define, and the middle needle position was
    # never where retrieval broke.
    for m in models:
        sh(["bench_longctx.py", m, "--depths", "16384", "65536", "131072",
            "--fracs", "0.1", "0.9",
            "--out", "longctx.jsonl"], "phase_longctx.log")


def phase_concurrency(models):
    # 1 / 4 / 8. np=2 sits on the straight part of the curve between 1 and 4.
    for m in models:
        sh(["bench_concurrency.py", m, "--levels", "1", "4", "8",
            "--n-predict", "256", "--out", "concurrency.jsonl"],
           "phase_concurrency.log")


PHASES = {
    "vram": phase_vram,
    "speed": phase_speed,
    "speed_reverse": phase_speed_reverse,
    "prefill": phase_prefill,
    "mtp": phase_mtp,
    "kvquant": phase_kvquant,
    "maxctx": phase_maxctx,
    "quality": phase_quality,
    "agentic": phase_agentic,
    "longctx": phase_longctx,
    "concurrency": phase_concurrency,
}

DEFAULT_ORDER = ["vram", "speed", "prefill", "mtp", "kvquant", "maxctx",
                 "agentic", "quality", "longctx", "concurrency"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", nargs="*", default=DEFAULT_ORDER)
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    args = ap.parse_args()

    t0 = time.time()
    for name in args.phases:
        if name not in PHASES:
            print(f"unknown phase {name}", flush=True)
            continue
        models = args.models
        if name == "mtp":
            # Only models whose GGUF carries a nextn block, and ornith-9b is
            # already fully measured on this axis in results_prep/ (both arms
            # run back-to-back, so the paired ratio stands). Re-running it would
            # be repeating a completed experiment.
            models = [m for m in models if m in ("ornith-9b", "ornith-35b-a3b", "qwen3.8-27b")]
        print(f"\n{'='*70}\nPHASE {name}  ({len(models)} models) "
              f"[{time.time()-t0:.0f}s elapsed]\n{'='*70}", flush=True)
        PHASES[name](models)
    print(f"\nALL PHASES DONE in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

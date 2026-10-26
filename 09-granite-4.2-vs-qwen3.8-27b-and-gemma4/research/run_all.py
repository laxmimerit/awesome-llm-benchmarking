"""Drive the whole sweep in a fixed order, in one session.

Order matters on this rig.  The card is power capped, so a ladder measured
cold reads 7-8% high; every speed ladder therefore runs forward and then again
in reverse as its own control, and no stage is compared against a number from
an earlier session.

Usage:
    python run_all.py --stage teardown
    python run_all.py --stage speed
    python run_all.py --stage quality
    python run_all.py --stage all
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from models import ALL, BRACKET_A, BRACKET_B, GRANITE, REGISTRY

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent / "logs"


def run(script, *args, log=None):
    cmd = [sys.executable, str(HERE / script), *[str(a) for a in args]]
    print(f"\n>>> {' '.join(cmd[1:])}", flush=True)
    t0 = time.time()
    if log:
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LOGS / log, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== {' '.join(cmd[1:])} =====\n")
            f.flush()
            p = subprocess.run(cmd, cwd=str(HERE), stdout=f,
                               stderr=subprocess.STDOUT)
    else:
        p = subprocess.run(cmd, cwd=str(HERE))
    print(f"<<< rc={p.returncode} in {time.time()-t0:.0f}s", flush=True)
    return p.returncode


def stage_teardown():
    args = [f"{k}={REGISTRY[k]['path']}" for k in ALL]
    run("teardown.py", *args, log="teardown.log")


def stage_speed():
    # forward, then the same ladder reversed as a drift control
    run("bench_speed.py", *ALL, "--out", "speed_fwd.jsonl", log="speed.log")
    run("bench_speed.py", *list(reversed(ALL)), "--out", "speed_rev.jsonl",
        log="speed.log")


def stage_vram():
    run("bench_vram.py", *ALL, "--out", "vram.jsonl", log="vram.log")


def stage_ctx():
    run("bench_usable_ctx.py", *ALL, "--out", "usable_ctx.jsonl", log="ctx.log")


def stage_concurrency():
    run("bench_concurrency.py", *ALL, "--out", "concurrency.jsonl",
        log="concurrency.log")


def stage_mtp():
    # only models whose GGUF carries a nextn block are worth testing
    run("check_mtp.py", "qwen3.8-27b", "ornith-9b", "--out", "mtp.jsonl",
        log="mtp.log")


def stage_quality():
    run("bench_hard.py", *ALL, "--seeds", 11, 22, "--out", "quality_hard.jsonl",
        log="quality.log")


def stage_code():
    run("bench_code.py", *BRACKET_A, "--seeds", 11, 22, "--out", "code.jsonl",
        log="code.log")
    run("bench_code.py", *BRACKET_B, "--seeds", 11, 22, "--out", "code.jsonl",
        log="code.log")


def stage_agentic():
    run("bench_agentic_hard.py", *ALL, "--seeds", 11, "--out",
        "agentic_hard.jsonl", log="agentic.log")


def stage_multihop():
    run("bench_multihop.py", *ALL, "--out", "multihop.jsonl", log="multihop.log")


def stage_effort():
    """Granite's effort dial, measured rather than believed."""
    for eff in ["low"]:
        run("bench_hard.py", *GRANITE, "--seeds", 11, "--effort", eff,
            "--out", "effort.jsonl", log="effort.log")
    run("bench_hard.py", *GRANITE, "--seeds", 11, "--out", "effort.jsonl",
        log="effort.log")


STAGES = {
    "teardown": stage_teardown,
    "vram": stage_vram,
    "speed": stage_speed,
    "mtp": stage_mtp,
    "ctx": stage_ctx,
    "concurrency": stage_concurrency,
    "quality": stage_quality,
    "code": stage_code,
    "agentic": stage_agentic,
    "multihop": stage_multihop,
    "effort": stage_effort,
}

ORDER = ["teardown", "vram", "speed", "mtp", "ctx", "concurrency",
         "quality", "code", "agentic", "multihop", "effort"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["all"])
    args = ap.parse_args()
    stages = ORDER if args.stage == ["all"] else args.stage
    for s in stages:
        print(f"\n########## STAGE {s} ##########", flush=True)
        STAGES[s]()


if __name__ == "__main__":
    main()

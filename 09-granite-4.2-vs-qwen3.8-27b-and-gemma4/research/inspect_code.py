"""Re-run one coding task on one model and show the code plus which asserts fail.

A pass/fail column cannot distinguish "the model misread the spec" from "my
harness mangled the extraction", and the second would be my bug, not a result.
This prints the extracted code and the exact failing assertions.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import server
from bench_code import split_asserts
from models import REGISTRY
from problems_code import CODE_INSTR, TASKS, extract_code

RESULTS = Path(__file__).resolve().parent.parent / "results"


def run_each(code, tests_src, timeout=20):
    stmts = split_asserts(tests_src)
    out = []
    with tempfile.TemporaryDirectory() as td:
        for s in stmts:
            f = Path(td) / "c.py"
            f.write_text(code + "\n\n" + s, encoding="utf-8")
            try:
                p = subprocess.run([sys.executable, "-I", str(f)],
                                   capture_output=True, text=True,
                                   timeout=timeout, cwd=td)
                ok = p.returncode == 0
                err = (p.stderr or "").strip().splitlines()[-1] if not ok else ""
            except subprocess.TimeoutExpired:
                ok, err = False, "timeout"
            out.append((ok, s.replace("\n", " ; ")[:110], err[:120]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--id", default="C6")
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--ctx", type=int, default=36864)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    task = next(t for t in TASKS if t["id"] == args.id)
    srv = server.Server(args.model, ctx=args.ctx, tag=f"{args.model}_inspect")
    srv.start()
    try:
        out = server.chat([{"role": "user", "content": task["prompt"] + CODE_INSTR}],
                          n_predict=args.max_tokens,
                          sampling=dict(REGISTRY[args.model]["sampling"]),
                          seed=args.seed)
    finally:
        srv.stop()

    msg = out["choices"][0]["message"]
    content = msg.get("content") or ""
    if "</think>" in content:
        content = content.split("</think>")[-1]
    code = extract_code(content)

    print(f"model={args.model} task={args.id} "
          f"tokens={out.get('usage', {}).get('completion_tokens')} "
          f"finish={out['choices'][0].get('finish_reason')}")
    print("\n--- extracted code ---")
    print(code)
    print("\n--- per assert ---")
    results = run_each(code, task["tests"])
    for ok, stmt, err in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {stmt}")
        if not ok and err:
            print(f"          {err}")
    print(f"\n{sum(1 for r in results if r[0])}/{len(results)} asserts pass")

    if args.save:
        (RESULTS / args.save).write_text(json.dumps({
            "model": args.model, "task": args.id, "code": code,
            "results": [{"ok": o, "stmt": s, "err": e} for o, s, e in results],
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

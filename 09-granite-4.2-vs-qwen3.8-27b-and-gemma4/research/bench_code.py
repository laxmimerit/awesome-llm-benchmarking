"""Executable coding eval: generate, then actually run hidden tests.

The model sees only the specification.  Its code is written to a temp file
together with the hidden tests and run in a separate interpreter with a hard
timeout, so grading is pass/fail on real execution rather than on how the code
looks.

Partial credit is recorded as well as pass/fail: the runner reports how many of
the individual asserts passed, which is what separates "misread one edge case"
from "did not understand the task".
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import server
from models import REGISTRY
from problems_code import CODE_INSTR, TASKS, extract_code

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Each assert is run independently so one early failure does not hide the rest.
HARNESS = '''
import json, sys
_results = []
for _i, _line in enumerate(_TESTS):
    try:
        exec(compile(_line, "<test>", "exec"), globals())
        _results.append(True)
    except Exception:
        _results.append(False)
print("__RESULT__" + json.dumps(_results))
'''


def split_asserts(tests_src):
    """Split the test block into independently runnable statements."""
    out, buf = [], []
    for line in tests_src.strip().splitlines():
        if not line.strip():
            continue
        if line.startswith((" ", "\t")) or line.startswith(("except", "else", "elif", "finally")):
            buf.append(line)
            continue
        if buf:
            out.append("\n".join(buf))
            buf = []
        buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out


def run_tests(code, tests_src, timeout=20):
    stmts = split_asserts(tests_src)
    src = (code + "\n\n_TESTS = " + repr(stmts) + "\n" + HARNESS)
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "candidate.py"
        f.write_text(src, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(f)],
                               capture_output=True, text=True, timeout=timeout,
                               cwd=td)
        except subprocess.TimeoutExpired:
            return {"passed": 0, "total": len(stmts), "ok": False,
                    "error": "timeout"}
        for line in p.stdout.splitlines():
            if line.startswith("__RESULT__"):
                res = json.loads(line[len("__RESULT__"):])
                return {"passed": sum(res), "total": len(res),
                        "ok": all(res), "error": None}
        return {"passed": 0, "total": len(stmts), "ok": False,
                "error": (p.stderr or p.stdout or "no output")[-300:]}


def run_task(task, sampling, max_tokens, seed, effort=None):
    kw = {}
    if effort is not None:
        kw["chat_template_kwargs"] = {"reasoning_effort": effort}
    out = server.chat([{"role": "user", "content": task["prompt"] + CODE_INSTR}],
                      n_predict=max_tokens, sampling=dict(sampling), seed=seed,
                      **kw)
    choice = out["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if not reasoning and "</think>" in content:
        head, _, tail = content.partition("</think>")
        reasoning, content = head.replace("<think>", ""), tail
    usage = out.get("usage", {})

    code = extract_code(content)
    verdict = run_tests(code, task["tests"])
    return {
        "id": task["id"],
        "tier": task["tier"],
        "func": task["func"],
        "correct": verdict["ok"],
        "asserts_passed": verdict["passed"],
        "asserts_total": verdict["total"],
        "assert_frac": round(verdict["passed"] / max(1, verdict["total"]), 3),
        "exec_error": verdict["error"],
        "code_chars": len(code),
        "had_fence": "```" in content,
        "finish_reason": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length",
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "wall_s": round(out["_wall_s"], 2),
        "seed": seed,
        "effort": effort,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="*", default=[11, 22])
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--sampling-json", default=None,
                    help="JSON object overriding the model's card sampling")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", default="code.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        m = REGISTRY[key]
        sampling = dict(m["sampling"])
        if args.sampling_json:
            sampling = json.loads(args.sampling_json)
        print(f"\n=== {key} code eval effort={args.effort} ===", flush=True)
        srv = server.Server(key, ctx=args.ctx, tag=f"{key}_code")
        srv.start()
        rows = []
        try:
            for seed in args.seeds:
                for task in TASKS:
                    t0 = time.time()
                    try:
                        r = run_task(task, sampling, args.max_tokens, seed,
                                     args.effort)
                    except Exception as e:
                        r = {"id": task["id"], "tier": task["tier"],
                             "correct": False, "error": str(e)[:300],
                             "seed": seed, "effort": args.effort}
                    r.update({"model": key, "sampling": sampling})
                    rows.append(r)
                    flag = "OK " if r.get("correct") else "XX "
                    print(f"  {flag}{task['id']:3} seed={seed} "
                          f"{r.get('asserts_passed','?')}/{r.get('asserts_total','?')} "
                          f"tok={str(r.get('completion_tokens')):>6} "
                          f"{time.time()-t0:6.1f}s"
                          f"{' TRUNC' if r.get('truncated') else ''}"
                          f"{' ERR:' + str(r.get('exec_error'))[:40] if r.get('exec_error') else ''}",
                          flush=True)
        finally:
            srv.stop()

        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  {key}: {sum(r.get('correct', False) for r in rows)}/{len(rows)} "
              f"tasks fully passed", flush=True)


if __name__ == "__main__":
    main()

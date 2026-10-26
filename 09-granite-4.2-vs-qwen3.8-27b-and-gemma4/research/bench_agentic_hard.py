"""Runner for the harder agentic suite.

Each task gets a fresh World so the retry trap in `open_manifest` resets, and
the turn budget is raised to 25 because the pagination and probe tasks
legitimately need more than ten calls.

Recorded per task: solved or not, tool calls made, malformed calls, whether the
model recovered from the deliberate manifest-key rejection, and whether it
exhausted the turn budget.
"""

import argparse
import json
import time
from pathlib import Path

import server
from agentic_hard import (ANSWER_INSTR, TASKS, TOOLS, World, dispatch, extract)
from models import REGISTRY

RESULTS = Path(__file__).resolve().parent.parent / "results"

SYSTEM = (
    "You are a careful assistant with tool access. Call the provided tools to "
    "gather facts before answering. Never invent a value you have not read "
    "from a tool result. If a tool returns an error, read the error and try "
    "again with corrected arguments."
)


def run_task(task, sampling, max_turns=25, max_tokens=8192, effort=None):
    world = World()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["prompt"] + ANSWER_INSTR}]
    n_calls = n_malformed = n_errors = 0
    completion_tokens = 0
    reasoning_chars = 0
    saw_reject = recovered = False
    transcript = []
    final_text = ""
    finish = "max_turns"

    kw = {}
    if effort is not None:
        kw["chat_template_kwargs"] = {"reasoning_effort": effort}

    for turn in range(max_turns):
        out = server.chat(msgs, n_predict=max_tokens, sampling=sampling,
                          tools=TOOLS, tool_choice="auto", **kw)
        choice = out["choices"][0]
        msg = choice["message"]
        completion_tokens += (out.get("usage") or {}).get("completion_tokens") or 0
        reasoning_chars += len(msg.get("reasoning_content") or "")
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        transcript.append({
            "turn": turn, "content": content[:800],
            "tool_calls": [{"name": tc["function"]["name"],
                            "arguments": str(tc["function"].get("arguments"))[:300]}
                           for tc in tool_calls]})

        if not tool_calls:
            final_text = content
            finish = choice.get("finish_reason") or "stop"
            break

        msgs.append({"role": "assistant", "content": content,
                     "tool_calls": tool_calls})
        for tc in tool_calls:
            n_calls += 1
            name = tc["function"]["name"]
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    raise ValueError("arguments not an object")
            except Exception as e:
                n_malformed += 1
                result = {"error": f"could not parse arguments: {e}",
                          "raw": str(raw)[:300]}
            else:
                result = dispatch(world, name, args)
                if isinstance(result, dict) and result.get("error"):
                    n_errors += 1
                    if "invalid manifest key" in str(result.get("error")):
                        saw_reject = True
                elif saw_reject and name == "open_manifest":
                    recovered = True
            msgs.append({"role": "tool",
                         "tool_call_id": tc.get("id", f"call_{n_calls}"),
                         "name": name,
                         "content": json.dumps(result)[:4000]})

    ans = extract(final_text)
    if ans is None:
        for t in reversed(transcript):
            ans = extract(t["content"])
            if ans:
                break

    return {
        "id": task["id"], "kind": task["kind"], "hops_expected": task["hops"],
        "correct": bool(ans is not None and task["checker"](ans)),
        "answer_raw": (ans or "")[:200],
        "tool_calls": n_calls,
        "malformed_calls": n_malformed,
        "tool_errors": n_errors,
        "hit_reject": saw_reject,
        "recovered_from_reject": recovered,
        "finish": finish,
        "exhausted_turns": finish == "max_turns",
        "completion_tokens": completion_tokens,
        "reasoning_chars": reasoning_chars,
        "transcript": transcript,
        "effort": effort,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="*", default=[11])
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", default="agentic_hard.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        m = REGISTRY[key]
        sampling = dict(m["sampling"])
        print(f"\n=== {key} agentic-hard ===", flush=True)
        srv = server.Server(key, ctx=args.ctx, tag=f"{key}_agentic")
        srv.start()
        rows = []
        try:
            for seed in args.seeds:
                for task in TASKS:
                    t0 = time.time()
                    try:
                        r = run_task(task, dict(sampling, seed=seed),
                                     args.max_turns, args.max_tokens,
                                     args.effort)
                    except Exception as e:
                        r = {"id": task["id"], "kind": task["kind"],
                             "correct": False, "error": str(e)[:300]}
                    r.update({"model": key, "seed": seed})
                    rows.append(r)
                    print(f"  {'OK ' if r.get('correct') else 'XX '}"
                          f"{task['id']:3} {task['kind']:18} "
                          f"calls={r.get('tool_calls')} "
                          f"malformed={r.get('malformed_calls')} "
                          f"tok={r.get('completion_tokens')} "
                          f"{time.time()-t0:6.1f}s"
                          f"{' EXHAUSTED' if r.get('exhausted_turns') else ''}",
                          flush=True)
        finally:
            srv.stop()
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        solvable = [r for r in rows if r.get("kind") != "trap"]
        traps = [r for r in rows if r.get("kind") == "trap"]
        print(f"  {key}: solvable {sum(r.get('correct', False) for r in solvable)}/"
              f"{len(solvable)}, traps {sum(r.get('correct', False) for r in traps)}/"
              f"{len(traps)}", flush=True)


if __name__ == "__main__":
    main()

"""Agentic tool-calling eval: does the model chain tools and land the answer?

This is the axis the Ornith 35B card makes its loudest claim on ("outperforms
Gemma 4-31B and Muse Glimmer-30B by wide margins on agentic coding"), so it is
worth measuring rather than repeating.

Recorded per task: solved or not, how many tool calls it took, how many calls
were malformed, and whether the run died by exhausting its turn budget.
"""

import argparse
import json
import time
from pathlib import Path

import server
from agentic_env import call_tool
from agentic_tasks import TASKS, extract
from models import REGISTRY

RESULTS = Path(__file__).resolve().parent.parent / "results"

SYSTEM = (
    "You are a careful assistant with tool access. Call the provided tools to "
    "gather facts before answering. Never invent a value you have not read "
    "from a tool result."
)


def run_task(task, sampling, max_turns=10, max_tokens=8192):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": task["prompt"]}]
    n_calls = 0
    n_malformed = 0
    completion_tokens = 0
    reasoning_chars = 0
    transcript = []
    final_text = ""
    finish = "max_turns"

    for turn in range(max_turns):
        out = server.chat(msgs, n_predict=max_tokens, sampling=sampling,
                          tools=task["tools"], tool_choice="auto")
        choice = out["choices"][0]
        msg = choice["message"]
        completion_tokens += (out.get("usage") or {}).get("completion_tokens") or 0
        reasoning_chars += len(msg.get("reasoning_content") or "")
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        transcript.append({"turn": turn, "content": content[:1500],
                           "tool_calls": [
                               {"name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"][:500]}
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
                result = {"error": f"could not parse arguments: {e}", "raw": str(raw)[:300]}
            else:
                result = call_tool(name, args)
                if isinstance(result, dict) and "unknown tool" in str(result.get("error", "")):
                    n_malformed += 1
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", f"call_{n_calls}"),
                         "name": name, "content": json.dumps(result)[:4000]})

    ans = extract(final_text)
    # Some models put the final answer in the last assistant turn before a
    # trailing tool call; fall back to scanning the whole transcript.
    if ans is None:
        for t in reversed(transcript):
            ans = extract(t["content"])
            if ans:
                break

    try:
        ok = bool(ans is not None and task["checker"](ans))
    except Exception:
        ok = False

    return {
        "id": task["id"], "family": task["family"],
        "solved": ok, "answer": (ans or "")[:300], "expected": task["expected"],
        "n_tool_calls": n_calls, "n_malformed": n_malformed,
        "used_enough_calls": n_calls >= task["min_calls"],
        "turns": len(transcript), "finish": finish,
        "completion_tokens": completion_tokens,
        "reasoning_chars": reasoning_chars,
        "transcript": transcript,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="*", default=[3, 5])
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="agentic.jsonl")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out

    for key in args.models:
        sampling = dict(REGISTRY[key]["sampling"])
        print(f"\n=== {key} agentic ===", flush=True)
        srv = server.Server(key, ctx=args.ctx, tag=f"{key}_agentic")
        srv.start()
        rows = []
        try:
            for seed in args.seeds:
                s = dict(sampling)
                s["seed"] = seed
                for task in TASKS:
                    t0 = time.time()
                    try:
                        r = run_task(task, s, args.max_turns, args.max_tokens)
                    except Exception as e:
                        r = {"id": task["id"], "family": task["family"],
                             "solved": False, "error": str(e)[:400],
                             "n_tool_calls": 0, "turns": 0}
                    r.update({"model": key, "seed": seed,
                              "wall_s": round(time.time() - t0, 1)})
                    rows.append(r)
                    mark = "OK " if r.get("solved") else "XX "
                    print(f"  {mark}{task['id']:22} seed={seed} "
                          f"calls={r.get('n_tool_calls')} "
                          f"malformed={r.get('n_malformed')} "
                          f"turns={r.get('turns')} {r['wall_s']:6.1f}s "
                          f"-> {str(r.get('answer'))[:60]!r}", flush=True)
        finally:
            srv.stop()
        with open(out_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        n_ok = sum(bool(r.get("solved")) for r in rows)
        print(f"  {key}: {n_ok}/{len(rows)} solved", flush=True)


if __name__ == "__main__":
    main()

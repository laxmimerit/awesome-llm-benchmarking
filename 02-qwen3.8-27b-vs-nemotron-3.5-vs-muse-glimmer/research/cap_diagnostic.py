"""Supplementary diagnostic: does a bigger budget rescue the capped tasks?

Three of the five failures in the main run ended with done_reason == "length",
meaning the model was still reasoning when it hit the 32,768-token ceiling.
That is ambiguous: it could be a model that cannot solve the task, or a model
that simply needed more room.

This script re-runs ONLY those model/task pairs with num_predict raised to
60,000. Prompts for these three tasks are a few hundred tokens, so 60,000
generated tokens still fits inside the same 65,536 context - VRAM and context
are unchanged, only the generation ceiling moves.

This is NOT part of the headline comparison. The headline stands at an equal
32,768-token budget for all three models. This only labels each capped failure
as "solved with more room" or "still wrong".
"""

import json

import hardbench2 as H

PAIRS = [
    ("nemotron", "repunit_digit_sum"),
    ("nemotron", "code_calc2"),
    ("glimmer", "tilings_4x10"),
]
BIG_PREDICT = 60000
OUT = H.BASE + "/research/results-capdiag.json"

TASK_BY_ID = {t["id"]: t for t in H.TASKS}

# group by model so each model loads exactly once
by_model = {}
for key, task_id in PAIRS:
    by_model.setdefault(key, []).append(task_id)

results = {}
for key, task_ids in by_model.items():
    model = H.MODELS[key]
    print(f"\n--- preparing {key}: unloading everything first ---", flush=True)
    H.wait_until_unloaded()
    print(f"=== {model} (num_predict={BIG_PREDICT}) ===", flush=True)
    for task_id in task_ids:
        task = TASK_BY_ID[task_id]
        r = H.api(
            "/api/chat",
            {
                "model": model,
                "messages": [{"role": "user", "content": task["prompt"]}],
                "stream": False,
                "options": {
                    "num_ctx": H.NUM_CTX,
                    "temperature": H.TEMPERATURE,
                    "num_predict": BIG_PREDICT,
                },
            },
        )
        content = r.get("message", {}).get("content", "")
        done_reason = r.get("done_reason", "")
        ok = H.grade(task, content)
        rec = {
            "task": task_id,
            "correct": ok,
            "capped_again": done_reason == "length",
            "done_reason": done_reason,
            "gen_tokens": r.get("eval_count", 0),
            "response_tail": content[-1500:],
        }
        results.setdefault(key, []).append(rec)
        verdict = (
            "SOLVED with more room" if ok
            else ("STILL CAPPED at 60k" if rec["capped_again"]
                  else "finished but STILL WRONG")
        )
        print(
            f"  {task_id}: {verdict} "
            f"(gen={rec['gen_tokens']}tok, done_reason={done_reason})",
            flush=True,
        )

H.wait_until_unloaded()
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("\nsaved", OUT)

"""Hard-problem benchmark for 4 local models (Ollama, 64K context).

Every task has an objectively verifiable answer. Code tasks are executed in
a subprocess with a timeout so a wrong-complexity solution fails instead of
hanging the run. Before each model loads, all other models are unloaded and
the script waits until /api/ps reports nothing running.

Ground truths were hand-verified before inclusion:
  digit sum 342, 3^27 mod 100 = 87, E[rolls to two straight sixes] = 42,
  2x12 domino tilings = 233, logic grid has a unique solution
  (Meera->fish, Rahul->Kolkata), multi-hop needle 214*3+7 = 649.
"""

import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

OLLAMA = "http://localhost:11434"
NUM_CTX = 65536
MODELS = {
    "nemotron": "nemotron-3.5-lightning:latest",
    "glimmer": "muse-glimmer:latest",
    "qwen": "qwen3.6:27b",
    "gemma": "gemma4:26b",
}
RESULTS_PATH = (
    "D:/Claude AI/AI Model Performance/nemotron-3.5-vs-muse-glimmer/"
    "research/results-hard.json"
)

# ---------------------------------------------------------------- helpers


def api(path, payload=None, timeout=2400):
    if payload is None:
        req = urllib.request.Request(OLLAMA + path)
    else:
        req = urllib.request.Request(
            OLLAMA + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


def final_answer(text):
    """Pull the value after the last 'Answer:' marker."""
    matches = re.findall(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()[-200:]


def answer_number_is(text, expected):
    ans = final_answer(text).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    return bool(nums) and any(
        abs(float(n) - expected) < 1e-6 for n in nums
    )


def strip_code_fences(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m[-1] if m else text


def run_code_test(model_code, test_block, timeout=30):
    """Run model code + test block in a subprocess. PASS on stdout wins."""
    src = model_code + "\n\n" + test_block
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return "PASS" in p.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------- long doc


def make_multihop_prompt():
    filler = [
        "The library committee approved new reading lamps for the study hall.",
        "Morning walkers reported that the park gate now opens at five thirty.",
        "The annual food festival will move to the riverside ground this year.",
        "Bus route 17 added two extra services during college exam weeks.",
        "The old clock tower was repainted by volunteers over the weekend.",
        "A pottery workshop for beginners starts next month at the art center.",
        "The vegetable market shifted to the new covered shed near the canal.",
        "Street lights on the east lane were upgraded to LED fixtures.",
        "The swimming pool will close for cleaning on the first Monday.",
        "A local team won the district quiz championship for the second time.",
    ]
    parts = []
    total_words = 0
    i = 0
    fact_a_done = False
    fact_b_done = False
    while total_words < 30000:
        s = filler[i % len(filler)]
        parts.append(f"Note {i + 1}: {s}")
        total_words += len(s.split()) + 2
        i += 1
        if total_words > 6000 and not fact_a_done:
            parts.append("Note A: Ravi's locker number at the club is 214.")
            fact_a_done = True
        if total_words > 24000 and not fact_b_done:
            parts.append(
                "Note B: The storage room code equals Ravi's locker number "
                "multiplied by 3, plus 7."
            )
            fact_b_done = True
    doc = "\n".join(parts)
    return (
        "Here is a long list of community notes.\n\n" + doc + "\n\n"
        "Question: Work out the storage room code. You will need to combine "
        "two separate notes. End your reply with 'Answer: <number>'."
    )


# ---------------------------------------------------------------- graders


def grade_output_prediction(resp):
    return (
        "[2, 2, 2]" in resp
        and "[1, 2] [1, 2]" in resp
        and re.search(r"\bTrue\b", resp)
    )


def grade_instruction(resp):
    text = resp.strip()
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]
    if len(sentences) != 5:
        return False
    for sent, letter in zip(sentences, "ABCDE"):
        if not sent[0].upper() == letter:
            return False
        words = sent.rstrip(".").split()
        if len(words) != 8:
            return False
    return True


def grade_json_hard(resp):
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        items = obj["items"]
        total = obj.get("total") or obj.get("total_price")
        drone = next(i for i in items if "falcon" in str(i["name"]).lower())
        battery = next(i for i in items if "batter" in str(i["name"]).lower())
        return (
            int(obj["order_id"]) == 4821
            and str(obj["date"]) == "2026-03-05"
            and str(obj["customer"]["name"]).lower() == "rohan sharma"
            and int(drone["quantity"]) == 2
            and float(drone["unit_price"]) == 45000
            and int(battery["quantity"]) == 1
            and float(total) == 93500
        )
    except Exception:
        return False


CALC_TEST = """
import math
cases = [
    ("2+3*4", 14), ("(2+3)*4", 20), ("10/4", 2.5),
    ("2*(3+(4-1))/3", 4), ("-3+5", 2), (" 12 * ( 2 + 1 ) ", 36),
    ("100/(2*5)/2", 5), ("7-2-3", 2),
]
for expr, want in cases:
    got = calc(expr)
    assert math.isclose(float(got), want), (expr, got, want)
print("PASS")
"""

LIS_TEST = """
import random, time
from bisect import bisect_left

def ref(nums):
    tails = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)

random.seed(7)
for n in [0, 1, 20, 200, 1000]:
    a = [random.randint(-1000, 1000) for _ in range(n)]
    assert lis_length(a) == ref(a), n
big = [random.randint(0, 10**9) for _ in range(200000)]
t0 = time.time()
got = lis_length(big)
elapsed = time.time() - t0
assert got == ref(big)
assert elapsed < 5.0, f"too slow: {elapsed}"
print("PASS")
"""

BUGFIX_TEST = """
r1 = add_item("apple")
r2 = add_item("banana")
assert r1 == {"apple": 1}, "default dict is still shared"
assert "apple" not in r2, "default dict is still shared"
assert r2 == {"banana": 1}
inv = {"pen": 2}
r3 = add_item("pen", inv)
assert r3 == {"pen": 3}
print("PASS")
"""

TASKS = [
    {
        "id": "aime_digit_sum",
        "prompt": (
            "Let N = 9 + 99 + 999 + 9999 + ... where the final term is made "
            "of 321 nines. Find the sum of the digits of N. End your reply "
            "with 'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 342),
    },
    {
        "id": "modular_arithmetic",
        "prompt": (
            "What are the last two digits of 3^27? End your reply with "
            "'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 87),
    },
    {
        "id": "expected_value",
        "prompt": (
            "A fair six-sided die is rolled repeatedly. What is the expected "
            "number of rolls until you first see two sixes in a row? End "
            "your reply with 'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 42),
    },
    {
        "id": "combinatorics",
        "prompt": (
            "In how many different ways can you completely tile a 2 x 12 "
            "board using 1 x 2 dominoes? End your reply with "
            "'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 233),
    },
    {
        "id": "logic_grid",
        "prompt": (
            "Four friends (Priya, Rahul, Meera, Arjun) each own a different "
            "pet (cat, dog, parrot, fish) and live in different cities "
            "(Delhi, Mumbai, Chennai, Kolkata).\n"
            "1. Priya does not live in Delhi.\n"
            "2. The dog owner lives in Mumbai.\n"
            "3. Rahul owns the parrot.\n"
            "4. Meera lives in Chennai.\n"
            "5. The cat owner lives in Delhi.\n"
            "6. Arjun does not own the dog.\n"
            "Who owns the fish, and in which city does Rahul live? End your "
            "reply with 'Answer: <fish owner>, <Rahul's city>'."
        ),
        "check": lambda r: (
            "meera" in final_answer(r).lower()
            and "kolkata" in final_answer(r).lower()
        ),
    },
    {
        "id": "code_expression_eval",
        "prompt": (
            "Write a Python function calc(expr) that evaluates an arithmetic "
            "expression given as a string. It must support +, -, *, /, "
            "parentheses, unary minus, spaces, and standard operator "
            "precedence, and use float division. Do NOT use eval or exec. "
            "Reply with only the code."
        ),
        "check": ("code", CALC_TEST),
    },
    {
        "id": "code_lis_nlogn",
        "prompt": (
            "Write a Python function lis_length(nums) that returns the "
            "length of the longest strictly increasing subsequence. It must "
            "run in O(n log n) time and handle 200,000 elements in well "
            "under 5 seconds. Reply with only the code."
        ),
        "check": ("code", LIS_TEST),
    },
    {
        "id": "code_output_prediction",
        "prompt": (
            "What exactly does this Python program print? Give the exact "
            "output lines.\n\n"
            "```python\n"
            "funcs = []\n"
            "for i in range(3):\n"
            "    funcs.append(lambda: i)\n"
            "def f(x, acc=[]):\n"
            "    acc.append(x)\n"
            "    return acc\n"
            "print([g() for g in funcs])\n"
            "print(f(1), f(2))\n"
            "a = f(3)\n"
            "print(a is f(4))\n"
            "```"
        ),
        "check": lambda r: bool(grade_output_prediction(r)),
    },
    {
        "id": "code_subtle_bugfix",
        "prompt": (
            "This function has a subtle bug that shows up only across "
            "multiple calls. Fix it. Each call without an explicit inventory "
            "must start from an empty inventory. Reply with only the "
            "corrected code.\n\n"
            "```python\n"
            "def add_item(item, inventory={}):\n"
            "    if item in inventory:\n"
            "        inventory[item] += 1\n"
            "    else:\n"
            "        inventory[item] = 1\n"
            "    return inventory\n"
            "```"
        ),
        "check": ("code", BUGFIX_TEST),
    },
    {
        "id": "instruction_brutal",
        "prompt": (
            "Write exactly 5 sentences about the ocean. The first sentence "
            "must start with a word beginning with A, the second with B, the "
            "third with C, the fourth with D, and the fifth with E. Every "
            "sentence must contain exactly 8 words and end with a period. "
            "Write nothing else, no preamble."
        ),
        "check": lambda r: bool(grade_instruction(r)),
    },
    {
        "id": "json_hard",
        "prompt": (
            "Convert this into JSON with exactly this structure: "
            '{"order_id": int, "date": "YYYY-MM-DD", '
            '"customer": {"name": str, "email": str}, '
            '"items": [{"name": str, "quantity": int, "unit_price": '
            'number}], "shipping_city": str, "total": number}. '
            "Compute total yourself. Output only the JSON.\n\n"
            "Order #4821 was placed on March 5th, 2026 by Rohan Sharma "
            "(rohan.s@example.com). He bought 2 units of the Falcon X1 "
            "drone at 45,000 rupees each and 1 spare battery at 3,500 "
            "rupees. Ship to: 12 MG Road, Bengaluru 560001."
        ),
        "check": lambda r: bool(grade_json_hard(r)),
    },
    {
        "id": "multihop_needle_64k",
        "prompt": make_multihop_prompt(),
        "check": lambda r: answer_number_is(r, 649),
    },
]


def grade(task, response):
    check = task["check"]
    if isinstance(check, tuple) and check[0] == "code":
        return run_code_test(strip_code_fences(response), check[1])
    try:
        return bool(check(response))
    except Exception:
        return False


# ---------------------------------------------------------------- runner


def wait_until_unloaded(max_wait=120):
    """Unload everything and wait until /api/ps reports no running models."""
    for model in MODELS.values():
        try:
            api("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            ps = api("/api/ps")
            if not ps.get("models"):
                print("  (gpu clear)", flush=True)
                return True
        except Exception:
            pass
        time.sleep(2)
    print("  WARNING: models still loaded after unload wait", flush=True)
    return False


def vram_of(model):
    try:
        ps = api("/api/ps")
        for m in ps.get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return round(m.get("size_vram", 0) / 1e9, 1)
    except Exception:
        pass
    return None


def run_model(key, model):
    results = []
    print(f"\n=== {model} ===", flush=True)
    vram = None
    for task in TASKS:
        t0 = time.time()
        try:
            r = api(
                "/api/chat",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": task["prompt"]}],
                    "stream": False,
                    "options": {
                        "num_ctx": NUM_CTX,
                        "temperature": 0.2,
                        "num_predict": 16384,
                    },
                },
            )
        except Exception as e:
            print(f"  {task['id']}: REQUEST FAILED {e}", flush=True)
            results.append({"task": task["id"], "correct": False,
                            "error": str(e)})
            continue
        wall = time.time() - t0
        if vram is None:
            vram = vram_of(model)
        msg = r.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", "")
        eval_count = r.get("eval_count", 0)
        eval_dur = r.get("eval_duration", 1) / 1e9
        p_count = r.get("prompt_eval_count", 0)
        p_dur = r.get("prompt_eval_duration", 1) / 1e9
        ok = grade(task, content)
        results.append(
            {
                "task": task["id"],
                "correct": ok,
                "response": content[-3000:],
                "thinking_words": len(thinking.split()) if thinking else 0,
                "gen_tokens": eval_count,
                "gen_tps": round(eval_count / eval_dur, 1) if eval_dur else 0,
                "prompt_tokens": p_count,
                "prompt_tps": round(p_count / p_dur, 1) if p_dur else 0,
                "wall_seconds": round(wall, 1),
            }
        )
        print(
            f"  {task['id']}: correct={ok} gen={eval_count}tok "
            f"@{results[-1]['gen_tps']}tps wall={wall:.0f}s",
            flush=True,
        )
    return {"vram_gb": vram, "tasks": results}


def main():
    try:
        with open(RESULTS_PATH, encoding="utf-8") as f:
            out = json.load(f)
    except Exception:
        out = {}
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS)
    for key in keys:
        model = MODELS[key]
        print(f"\n--- preparing {key}: unloading everything first ---",
              flush=True)
        wait_until_unloaded()
        out[key] = run_model(key, model)
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        n_ok = sum(1 for t in out[key]["tasks"] if t["correct"])
        print(f"  >> {key}: {n_ok}/{len(TASKS)} correct, saved", flush=True)
    wait_until_unloaded()
    print("\nAll done.")


if __name__ == "__main__":
    main()

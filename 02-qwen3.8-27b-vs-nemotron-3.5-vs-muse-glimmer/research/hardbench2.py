"""Hard-problem benchmark v2 for Qwen 3.8 27B vs Nemotron 3.5 Lightning vs
Muse Glimmer (Ollama, RTX 5090, 64K context).

Why v2: the previous 12-task suite saturated (Glimmer scored 12/12), so it
could no longer separate models. Every task here is harder, and each ground
truth is produced by exact computation in verify_truth.py / verify_tasks.py
rather than by hand.

Two methodology fixes over v1:
  1. num_predict raised 16384 -> 32768, because v1 recorded several failures
     that were really generation-cap exhaustion mid-reasoning.
  2. done_reason is recorded per task, so a cap-induced failure is reported
     separately from a genuine reasoning failure.

Exclusive loading: before any model loads, EVERY model reported by /api/tags
is unloaded and /api/ps is polled until it reports nothing resident.
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
NUM_PREDICT = 32768
TEMPERATURE = 0.2

MODELS = {
    "qwen38": "qwen3.8:latest",   # 27.3B, Q4_K_M (the 27B build)
    "nemotron": "nemotron-3.5-lightning:latest",
    "glimmer": "muse-glimmer:latest",
}
BASE = "D:/Claude AI/AI Model Performance/qwen3.8-vs-nemotron-vs-glimmer"
RESULTS_PATH = BASE + "/research/results-hard2.json"

# ---------------------------------------------------------------- helpers


def api(path, payload=None, timeout=5400):
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


def answer_number_is(text, expected, tol=1e-6):
    ans = final_answer(text).replace(",", "").replace("$", "")
    # allow a fraction like 9/4
    for a, b in re.findall(r"(-?\d+)\s*/\s*(\d+)", ans):
        try:
            if abs(float(a) / float(b) - expected) < tol:
                return True
        except ZeroDivisionError:
            pass
    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    return bool(nums) and any(abs(float(n) - expected) < tol for n in nums)


def strip_code_fences(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m[-1] if m else text


def run_code_test(model_code, test_block, timeout=60, forbid=None):
    """Run model code + test block in a subprocess. PASS on stdout wins.

    `forbid` enforces prompt rules that execution cannot check on its own:
    an eval-based calculator computes the right answers, so the only way to
    enforce "do not use eval" is to reject the submission statically.
    """
    if forbid:
        flat = re.sub(r"\s+", "", model_code)
        for bad in forbid:
            if re.sub(r"\s+", "", bad) in flat:
                return False
    src = model_code + "\n\n" + test_block
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run(
            [sys.executable, path], capture_output=True, text=True,
            timeout=timeout,
        )
        return "PASS" in p.stdout
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ---------------------------------------------------------------- long doc


def make_multihop_prompt():
    """~40k words of plausible filler with THREE chained facts and decoys."""
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
        "The community hall booking desk now accepts online payments.",
        "Cycling club members planted saplings along the lake path.",
    ]
    # Decoys: same shape as the real facts, different people and numbers.
    decoys = [
        "Note: Sunil's locker number at the club is 118.",
        "Note: Priya's locker number at the gym is 305.",
        "Note: The old storage room code was retired last year.",
        "Note: The archive shelf identifier is unrelated to any locker.",
    ]
    parts = []
    total_words = 0
    i = 0
    placed = {"a": False, "b": False, "c": False}
    d = 0
    while total_words < 40000:
        s = filler[i % len(filler)]
        parts.append(f"Note {i + 1}: {s}")
        total_words += len(s.split()) + 2
        i += 1
        if i % 40 == 0 and d < len(decoys):
            parts.append(decoys[d])
            total_words += len(decoys[d].split())
            d += 1
        if total_words > 5000 and not placed["a"]:
            parts.append("Note A: Ravi's locker number at the club is 214.")
            placed["a"] = True
        if total_words > 20000 and not placed["b"]:
            parts.append(
                "Note B: The storage room code equals Ravi's locker number "
                "multiplied by 3, plus 7."
            )
            placed["b"] = True
        if total_words > 34000 and not placed["c"]:
            parts.append(
                "Note C: The shelf identifier equals the storage room code "
                "plus 128."
            )
            placed["c"] = True
    doc = "\n".join(parts)
    return (
        "Here is a long list of community notes.\n\n" + doc + "\n\n"
        "Question: Work out the shelf identifier. You will need to combine "
        "three separate notes, and some notes are deliberately misleading. "
        "End your reply with 'Answer: <number>'."
    )


# ---------------------------------------------------------------- graders


def grade_output_prediction(resp):
    """Expected exact output verified in verify_tasks.py:
    [(1, 2)] / {'a': 2, 'b': 1} / [1, 2, 3] [1, 2, 3, 4] / True
    """
    norm = resp.replace("'", "'").replace('"', "'")
    norm = re.sub(r"\s+", " ", norm)
    return (
        "[(1, 2)]" in norm
        and "{'a': 2, 'b': 1}" in norm
        and "[1, 2, 3] [1, 2, 3, 4]" in norm
        and re.search(r"\bTrue\b", norm) is not None
    )


def grade_instruction(resp):
    """6 sentences, initials spell PYTHON, exactly 7 words each."""
    text = resp.strip()
    text = re.sub(r"^```.*?\n|```$", "", text, flags=re.DOTALL)
    sentences = [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]
    if len(sentences) != 6:
        return False
    for sent, letter in zip(sentences, "PYTHON"):
        if not sent or sent[0].upper() != letter:
            return False
        if len(sent.rstrip(".").split()) != 7:
            return False
    return True


def grade_json_hard(resp):
    """Subtotal 32650, 10% discount 3265, 18% GST 5289.30, total 34674.30."""
    m = re.search(r"\{.*\}", resp, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        items = obj["items"]
        kb = next(i for i in items if "vertex" in str(i["name"]).lower())
        ms = next(i for i in items if "nova" in str(i["name"]).lower())

        def num(*keys):
            for k in keys:
                if k in obj:
                    return float(obj[k])
            raise KeyError(keys)

        return (
            int(obj["order_id"]) == 7734
            and str(obj["date"]) == "2026-07-19"
            and str(obj["customer"]["name"]).lower() == "ananya iyer"
            and int(kb["quantity"]) == 3
            and float(kb["unit_price"]) == 8750
            and int(ms["quantity"]) == 2
            and abs(num("subtotal") - 32650) < 0.01
            and abs(num("discount", "discount_amount") - 3265) < 0.01
            and abs(num("tax", "gst", "tax_amount") - 5289.30) < 0.01
            and abs(num("total", "grand_total") - 34674.30) < 0.01
        )
    except Exception:
        return False


# ------------------------------------------------------------ code tests

CALC2_TEST = """
import math
cases = [
    ("2+3*4", 14), ("(2+3)*4", 20), ("10/4", 2.5), ("7//2", 3),
    ("7%3", 1), ("2**3**2", 512), ("-3+5", 2), ("-2**2", -4),
    ("2*(3+(4-1))/3", 4), (" 12 * ( 2 + 1 ) ", 36),
    ("100/(2*5)/2", 5), ("7-2-3", 2), ("(-3)**2", 9),
    ("10%4*2", 4), ("2+3*4**2", 50),
]
for expr, want in cases:
    got = calc(expr)
    assert math.isclose(float(got), float(want)), (expr, got, want)
print("PASS")
"""

INVERSIONS_TEST = """
import random, time

def ref(a):
    import bisect
    sorted_list = []
    inv = 0
    for x in reversed(a):
        i = bisect.bisect_left(sorted_list, x)
        inv += i
        sorted_list.insert(i, x)
    return inv

random.seed(11)
for n in [0, 1, 2, 10, 200]:
    a = [random.randint(-50, 50) for _ in range(n)]
    assert count_inversions(list(a)) == ref(a), (n, a[:20])
assert count_inversions([1,2,3,4]) == 0
assert count_inversions([4,3,2,1]) == 6

big = [random.randint(0, 10**9) for _ in range(300000)]
t0 = time.time()
got = count_inversions(list(big))
elapsed = time.time() - t0
assert got == ref(big), (got,)
assert elapsed < 6.0, f"too slow: {elapsed}"
print("PASS")
"""

LRU_TEST = """
import time, random
c = LRUCache(2)
c.put(1, 1); c.put(2, 2)
assert c.get(1) == 1
c.put(3, 3)                 # evicts key 2 (least recently used)
assert c.get(2) == -1
c.put(4, 4)                 # evicts key 1
assert c.get(1) == -1
assert c.get(3) == 3
assert c.get(4) == 4

c2 = LRUCache(2)
c2.put(1, 1); c2.put(2, 2); c2.put(1, 10)   # update must refresh recency
c2.put(3, 3)
assert c2.get(2) == -1
assert c2.get(1) == 10

random.seed(3)
big = LRUCache(1000)
t0 = time.time()
for i in range(200000):
    big.put(random.randint(0, 5000), i)
    big.get(random.randint(0, 5000))
elapsed = time.time() - t0
assert elapsed < 6.0, f"too slow: {elapsed}"
print("PASS")
"""

CLOSURE_FIX_TEST = """
ms = make_multipliers([2, 3, 4])
assert [m(10) for m in ms] == [20, 30, 40], [m(10) for m in ms]
ms2 = make_multipliers([])
assert ms2 == [] or list(ms2) == []
ms3 = make_multipliers([5])
assert ms3[0](3) == 15
print("PASS")
"""

VERSION_TEST = """
cases = [
    ("1.0", "1.0.0", 0), ("1.0.1", "1.0", 1), ("1.2", "1.10", -1),
    ("01.1", "1.1", 0), ("1.0.0", "1", 0), ("2.1", "2.1.0.0", 0),
    ("1.0.33", "1.0.27", 1), ("0.1", "1.1", -1), ("1.1", "1.1", 0),
]
for a, b, want in cases:
    got = compare_versions(a, b)
    assert got == want, (a, b, got, want)
print("PASS")
"""

# ---------------------------------------------------------------- tasks

TASKS = [
    # ------------------------------------------------------------ math
    {
        "id": "repunit_digit_sum",
        "cat": "Math",
        "label": "Repunit digit sum",
        "prompt": (
            "Let N = 1 + 11 + 111 + 1111 + ... where the final term is a "
            "number made of 300 ones. Find the sum of the digits of N. "
            "End your reply with 'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 1230),
    },
    {
        "id": "power_tower_mod",
        "cat": "Math",
        "label": "7^(7^7) mod 1000",
        "prompt": (
            "What are the last three digits of 7^(7^7)? Note this is a power "
            "tower: the exponent is 7^7. End your reply with "
            "'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 343),
    },
    {
        "id": "expected_three_sixes",
        "cat": "Math",
        "label": "E[rolls to three 6s]",
        "prompt": (
            "A fair six-sided die is rolled repeatedly. What is the expected "
            "number of rolls until you first see three sixes in a row? "
            "End your reply with 'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 258),
    },
    {
        "id": "tilings_4x10",
        "cat": "Math",
        "label": "4x10 domino tilings",
        "prompt": (
            "In how many different ways can you completely tile a 4 x 10 "
            "board using 1 x 2 dominoes? End your reply with "
            "'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 18061),
    },
    {
        "id": "mississippi_no_ss",
        "cat": "Math",
        "label": "MISSISSIPPI, no two S",
        "prompt": (
            "How many distinct arrangements of the letters of the word "
            "MISSISSIPPI have no two S's adjacent to each other? "
            "End your reply with 'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 7350),
    },
    {
        "id": "expected_first_blue",
        "cat": "Math",
        "label": "E[draws to first blue]",
        "prompt": (
            "A bag contains 5 red balls and 3 blue balls. You draw balls one "
            "at a time without replacement until you draw a blue ball. What "
            "is the expected number of draws? Give the exact value. "
            "End your reply with 'Answer: <number>'."
        ),
        "check": lambda r: answer_number_is(r, 2.25, tol=1e-4),
    },
    # ------------------------------------------------------------ code
    {
        "id": "code_calc2",
        "cat": "Code",
        "label": "Expression evaluator",
        "prompt": (
            "Write a Python function calc(expr) that evaluates an arithmetic "
            "expression given as a string. It must support +, -, *, / (float "
            "division), // (floor division), % (modulo), ** (exponentiation, "
            "right-associative), parentheses, unary minus, and spaces, with "
            "standard Python operator precedence. Note that -2**2 must equal "
            "-4. Do NOT use eval, exec, ast, or any parsing library. "
            "Reply with only the code."
        ),
        "check": ("code", CALC2_TEST,
                  ["eval(", "exec(", "importast", "ast.", "compile("]),
    },
    {
        "id": "code_inversions",
        "cat": "Code",
        "label": "Count inversions O(n log n)",
        "prompt": (
            "Write a Python function count_inversions(nums) that returns the "
            "number of pairs (i, j) with i < j and nums[i] > nums[j]. It must "
            "run in O(n log n) time and handle 300,000 elements in well under "
            "6 seconds. Reply with only the code."
        ),
        "check": ("code", INVERSIONS_TEST),
    },
    {
        "id": "code_lru",
        "cat": "Code",
        "label": "O(1) LRU cache",
        "prompt": (
            "Write a Python class LRUCache with __init__(self, capacity), "
            "get(self, key) and put(self, key, value). get returns -1 if the "
            "key is absent. Both operations must be O(1) average time. "
            "Updating an existing key must also refresh its recency. When "
            "over capacity, evict the least recently used key. It must handle "
            "200,000 mixed operations in well under 6 seconds. "
            "Reply with only the code."
        ),
        "check": ("code", LRU_TEST),
    },
    {
        "id": "code_closure_bug",
        "cat": "Code",
        "label": "Closure late-binding fix",
        "prompt": (
            "This function is supposed to return a list of functions where "
            "the k-th function multiplies its argument by factors[k], but it "
            "does not work. Fix it. Reply with only the corrected code.\n\n"
            "```python\n"
            "def make_multipliers(factors):\n"
            "    result = []\n"
            "    for f in factors:\n"
            "        result.append(lambda x: x * f)\n"
            "    return result\n"
            "```"
        ),
        "check": ("code", CLOSURE_FIX_TEST),
    },
    {
        "id": "code_version_compare",
        "cat": "Code",
        "label": "Version comparison",
        "prompt": (
            "Write a Python function compare_versions(a, b) that compares two "
            "dotted version strings. Return 1 if a > b, -1 if a < b, and 0 if "
            "they are equal. Missing trailing components count as zero, so "
            "'1.0' equals '1.0.0'. Leading zeros are insignificant, so '01.1' "
            "equals '1.1'. Components are compared numerically, so '1.2' is "
            "less than '1.10'. Reply with only the code."
        ),
        "check": ("code", VERSION_TEST),
    },
    {
        "id": "code_output_prediction",
        "cat": "Code",
        "label": "Predict exact output",
        "prompt": (
            "What exactly does this Python program print? Give the exact "
            "output lines and nothing else.\n\n"
            "```python\n"
            "def gen():\n"
            "    yield 1\n"
            "    yield 2\n"
            "    yield 3\n"
            "\n"
            "g = gen()\n"
            "print(list(zip(g, g)))\n"
            "\n"
            "d = {}\n"
            "for k in ['a', 'b', 'a']:\n"
            "    d[k] = d.get(k, 0) + 1\n"
            "print(d)\n"
            "\n"
            "x = [1, 2, 3]\n"
            "y = x[:]\n"
            "y.append(4)\n"
            "print(x, y)\n"
            "\n"
            "print([i for i in range(3)] == list(range(3)))\n"
            "```"
        ),
        "check": lambda r: bool(grade_output_prediction(r)),
    },
    # ------------------------------------------------------- reasoning
    {
        "id": "logic_grid_5x5",
        "cat": "Reasoning",
        "label": "5x5 logic grid",
        "prompt": (
            "Five friends (Aditi, Bhavesh, Chitra, Dev, Esha) each own a "
            "different pet (cat, dog, parrot, fish, rabbit) and live in "
            "different cities (Delhi, Mumbai, Chennai, Kolkata, Pune).\n"
            "1. The dog owner lives in Mumbai.\n"
            "2. Chitra lives in Chennai.\n"
            "3. Bhavesh owns the parrot.\n"
            "4. The cat owner lives in Delhi.\n"
            "5. Aditi lives in neither Delhi nor Mumbai.\n"
            "6. Dev does not own the dog.\n"
            "7. The person living in Pune owns the rabbit.\n"
            "8. Esha does not live in Kolkata.\n"
            "Who owns the fish, and in which city does Aditi live? End your "
            "reply with 'Answer: <fish owner>, <Aditi's city>'."
        ),
        "check": lambda r: (
            "chitra" in final_answer(r).lower()
            and "pune" in final_answer(r).lower()
        ),
    },
    {
        "id": "instruction_pythonic",
        "cat": "Reasoning",
        "label": "PYTHON acrostic, 7 words",
        "prompt": (
            "Write exactly 6 sentences about mountains. The first letters of "
            "the six sentences must spell PYTHON: sentence 1 starts with a "
            "word beginning with P, sentence 2 with Y, sentence 3 with T, "
            "sentence 4 with H, sentence 5 with O, sentence 6 with N. Every "
            "sentence must contain exactly 7 words and end with a period. "
            "Write nothing else, no preamble, no numbering."
        ),
        "check": lambda r: bool(grade_instruction(r)),
    },
    {
        "id": "json_computed",
        "cat": "Reasoning",
        "label": "JSON with computed totals",
        "prompt": (
            "Convert this into JSON with exactly this structure: "
            '{"order_id": int, "date": "YYYY-MM-DD", '
            '"customer": {"name": str, "email": str}, '
            '"items": [{"name": str, "quantity": int, "unit_price": number}], '
            '"subtotal": number, "discount": number, "tax": number, '
            '"shipping_city": str, "total": number}. '
            "Compute subtotal, discount, tax and total yourself. "
            "Output only the JSON.\n\n"
            "Order #7734 was placed on July 19th, 2026 by Ananya Iyer "
            "(ananya.i@example.com). She bought 3 units of the Vertex Pro "
            "keyboard at 8,750 rupees each and 2 units of the Nova mouse at "
            "3,200 rupees each. A 10% discount applies to the subtotal, and "
            "18% GST is charged on the discounted amount. "
            "Ship to: 44 Banjara Hills, Hyderabad 500081."
        ),
        "check": lambda r: bool(grade_json_hard(r)),
    },
    {
        "id": "multihop_needle_3hop",
        "cat": "Reasoning",
        "label": "64K three-hop needle",
        "prompt": make_multihop_prompt(),
        "check": lambda r: answer_number_is(r, 777),
    },
]


def grade(task, response):
    check = task["check"]
    if isinstance(check, tuple) and check[0] == "code":
        forbid = check[2] if len(check) > 2 else None
        return run_code_test(
            strip_code_fences(response), check[1], forbid=forbid
        )
    try:
        return bool(check(response))
    except Exception:
        return False


# ---------------------------------------------------------------- runner


def all_installed():
    try:
        tags = api("/api/tags", timeout=60)
        return [m["name"] for m in tags.get("models", [])]
    except Exception:
        return list(MODELS.values())


def wait_until_unloaded(max_wait=180):
    """Unload EVERY installed model, then poll /api/ps until it is empty."""
    for model in all_installed():
        try:
            api("/api/generate", {"model": model, "keep_alive": 0}, timeout=60)
        except Exception:
            pass
    t0 = time.time()
    while time.time() - t0 < max_wait:
        try:
            ps = api("/api/ps", timeout=30)
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
        ps = api("/api/ps", timeout=30)
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
                        "temperature": TEMPERATURE,
                        "num_predict": NUM_PREDICT,
                    },
                },
            )
        except Exception as e:
            print(f"  {task['id']}: REQUEST FAILED {e}", flush=True)
            results.append(
                {"task": task["id"], "cat": task["cat"],
                 "label": task["label"], "correct": False, "error": str(e)}
            )
            continue
        wall = time.time() - t0
        if vram is None:
            vram = vram_of(model)
        msg = r.get("message", {})
        content = msg.get("content", "")
        thinking = msg.get("thinking", "") or ""
        eval_count = r.get("eval_count", 0)
        eval_dur = r.get("eval_duration", 1) / 1e9
        p_count = r.get("prompt_eval_count", 0)
        p_dur = r.get("prompt_eval_duration", 1) / 1e9
        done_reason = r.get("done_reason", "")
        capped = done_reason == "length"
        ok = grade(task, content)
        results.append(
            {
                "task": task["id"],
                "cat": task["cat"],
                "label": task["label"],
                "correct": ok,
                "capped": capped,
                "done_reason": done_reason,
                "response": content[-4000:],
                "thinking_words": len(thinking.split()),
                "gen_tokens": eval_count,
                "gen_tps": round(eval_count / eval_dur, 1) if eval_dur else 0,
                "prompt_tokens": p_count,
                "prompt_tps": round(p_count / p_dur, 1) if p_dur else 0,
                "wall_seconds": round(wall, 1),
            }
        )
        flag = " CAPPED" if capped else ""
        print(
            f"  {task['id']}: correct={ok} gen={eval_count}tok "
            f"@{results[-1]['gen_tps']}tps wall={wall:.0f}s{flag}",
            flush=True,
        )
    return {"model": model, "vram_gb": vram, "tasks": results}


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
        out[key]["config"] = {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        n_ok = sum(1 for t in out[key]["tasks"] if t["correct"])
        n_cap = sum(1 for t in out[key]["tasks"] if t.get("capped"))
        print(
            f"  >> {key}: {n_ok}/{len(TASKS)} correct "
            f"({n_cap} hit the generation cap), saved",
            flush=True,
        )
    wait_until_unloaded()
    print("\nAll done.")


if __name__ == "__main__":
    main()

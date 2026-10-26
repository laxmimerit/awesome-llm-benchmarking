"""The 12 hard tasks, ported verbatim from 4-model-hard-benchmark.

Ground truths were hand-verified in that run and are unchanged here:
  digit sum 342, 3^27 mod 100 = 87, E[rolls to two straight sixes] = 42,
  2x12 domino tilings = 233, logic grid has a unique solution
  (Meera->fish, Rahul->Kolkata), multi-hop needle 214*3+7 = 649.

The tasks and graders are byte-identical to that run, but the conditions are
NOT, so scores must not be compared across the two articles as if controlled:

  30B run   Ollama,          temperature 0.2, 64K ctx, 16384-token cap
  this run  llama.cpp server, temperature 1.0, top_p 0.95, top_k 20, varied cap

Same questions, different runtime and different sampling. Treat any
cross-article comparison as indicative only.
"""
import json
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------- helpers


def final_answer(text):
    """Pull the value after the last 'Answer:' marker."""
    matches = re.findall(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
    return matches[-1].strip() if matches else text.strip()[-200:]


def answer_number_is(text, expected):
    ans = final_answer(text).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", ans)
    return bool(nums) and any(abs(float(n) - expected) < 1e-6 for n in nums)


def strip_code_fences(text):
    m = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return m[-1] if m else text


def run_code_test(model_code, test_block, timeout=30):
    """Run model code + test block in a subprocess. PASS on stdout wins."""
    src = model_code + "\n\n" + test_block
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
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
        parts.append("Note %d: %s" % (i + 1, s))
        total_words += len(s.split()) + 2
        i += 1
        if total_words > 6000 and not fact_a_done:
            parts.append("Note A: Ravi's locker number at the club is 214.")
            fact_a_done = True
        if total_words > 24000 and not fact_b_done:
            parts.append("Note B: The storage room code equals Ravi's locker "
                         "number multiplied by 3, plus 7.")
            fact_b_done = True
    doc = "\n".join(parts)
    return ("Here is a long list of community notes.\n\n" + doc + "\n\n"
            "Question: Work out the storage room code. You will need to "
            "combine two separate notes. End your reply with "
            "'Answer: <number>'.")


# ---------------------------------------------------------------- graders


def grade_output_prediction(resp):
    return ("[2, 2, 2]" in resp
            and "[1, 2] [1, 2]" in resp
            and re.search(r"\bTrue\b", resp))


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
        return (int(obj["order_id"]) == 4821
                and str(obj["date"]) == "2026-03-05"
                and str(obj["customer"]["name"]).lower() == "rohan sharma"
                and int(drone["quantity"]) == 2
                and float(drone["unit_price"]) == 45000
                and int(battery["quantity"]) == 1
                and float(total) == 93500)
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
        "cat": "math",
        "prompt": ("Let N = 9 + 99 + 999 + 9999 + ... where the final term is "
                   "made of 321 nines. Find the sum of the digits of N. End "
                   "your reply with 'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 342),
    },
    {
        "id": "modular_arithmetic",
        "cat": "math",
        "prompt": ("What are the last two digits of 3^27? End your reply with "
                   "'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 87),
    },
    {
        "id": "expected_value",
        "cat": "math",
        "prompt": ("A fair six-sided die is rolled repeatedly. What is the "
                   "expected number of rolls until you first see two sixes in "
                   "a row? End your reply with 'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 42),
    },
    {
        "id": "combinatorics",
        "cat": "math",
        "prompt": ("In how many different ways can you completely tile a "
                   "2 x 12 board using 1 x 2 dominoes? End your reply with "
                   "'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 233),
    },
    {
        "id": "logic_grid",
        "cat": "logic",
        "prompt": ("Four friends (Priya, Rahul, Meera, Arjun) each own a "
                   "different pet (cat, dog, parrot, fish) and live in "
                   "different cities (Delhi, Mumbai, Chennai, Kolkata).\n"
                   "1. Priya does not live in Delhi.\n"
                   "2. The dog owner lives in Mumbai.\n"
                   "3. Rahul owns the parrot.\n"
                   "4. Meera lives in Chennai.\n"
                   "5. The cat owner lives in Delhi.\n"
                   "6. Arjun does not own the dog.\n"
                   "Who owns the fish, and in which city does Rahul live? End "
                   "your reply with 'Answer: <fish owner>, <Rahul's city>'."),
        "check": lambda r: ("meera" in final_answer(r).lower()
                            and "kolkata" in final_answer(r).lower()),
    },
    {
        "id": "code_expression_eval",
        "cat": "code",
        "prompt": ("Write a Python function calc(expr) that evaluates an "
                   "arithmetic expression given as a string. It must support "
                   "+, -, *, /, parentheses, unary minus, spaces, and standard "
                   "operator precedence, and use float division. Do NOT use "
                   "eval or exec. Reply with only the code."),
        "check": ("code", CALC_TEST),
    },
    {
        "id": "code_lis_nlogn",
        "cat": "code",
        "prompt": ("Write a Python function lis_length(nums) that returns the "
                   "length of the longest strictly increasing subsequence. It "
                   "must run in O(n log n) time and handle 200,000 elements in "
                   "well under 5 seconds. Reply with only the code."),
        "check": ("code", LIS_TEST),
    },
    {
        "id": "code_output_prediction",
        "cat": "code",
        "prompt": ("What exactly does this Python program print? Give the "
                   "exact output lines.\n\n"
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
                   "```"),
        "check": lambda r: bool(grade_output_prediction(r)),
    },
    {
        "id": "code_subtle_bugfix",
        "cat": "code",
        "prompt": ("This function has a subtle bug that shows up only across "
                   "multiple calls. Fix it. Each call without an explicit "
                   "inventory must start from an empty inventory. Reply with "
                   "only the corrected code.\n\n"
                   "```python\n"
                   "def add_item(item, inventory={}):\n"
                   "    if item in inventory:\n"
                   "        inventory[item] += 1\n"
                   "    else:\n"
                   "        inventory[item] = 1\n"
                   "    return inventory\n"
                   "```"),
        "check": ("code", BUGFIX_TEST),
    },
    {
        "id": "instruction_brutal",
        "cat": "instruct",
        "prompt": ("Write exactly 5 sentences about the ocean. The first "
                   "sentence must start with a word beginning with A, the "
                   "second with B, the third with C, the fourth with D, and "
                   "the fifth with E. Every sentence must contain exactly 8 "
                   "words and end with a period. Write nothing else, no "
                   "preamble."),
        "check": lambda r: bool(grade_instruction(r)),
    },
    {
        "id": "json_hard",
        "cat": "format",
        "prompt": ("Convert this into JSON with exactly this structure: "
                   '{"order_id": int, "date": "YYYY-MM-DD", '
                   '"customer": {"name": str, "email": str}, '
                   '"items": [{"name": str, "quantity": int, "unit_price": '
                   'number}], "shipping_city": str, "total": number}. '
                   "Compute total yourself. Output only the JSON.\n\n"
                   "Order #4821 was placed on March 5th, 2026 by Rohan Sharma "
                   "(rohan.s@example.com). He bought 2 units of the Falcon X1 "
                   "drone at 45,000 rupees each and 1 spare battery at 3,500 "
                   "rupees. Ship to: 12 MG Road, Bengaluru 560001."),
        "check": lambda r: bool(grade_json_hard(r)),
    },
    {
        "id": "multihop_needle_64k",
        "cat": "longctx",
        "prompt": make_multihop_prompt(),
        "check": lambda r: answer_number_is(r, 649),
    },
]


def grade(task, response):
    """Return (passed, detail). Never raises."""
    check = task["check"]
    if isinstance(check, tuple) and check[0] == "code":
        code = strip_code_fences(response)
        if not code.strip():
            return False, "no code emitted"
        return run_code_test(code, check[1]), "subprocess test"
    try:
        return bool(check(response)), "answer=%r" % final_answer(response)[:80]
    except Exception as e:
        return False, "grader error: %s" % str(e)[:60]


# ---------------------------------------------------------------- game task
# Scored structurally only. Real quality is judged by playing the file, so the
# raw HTML is written to games/ for manual review. Kept out of TASKS because it
# needs a much larger token budget than the 12 hard tasks.

GAME_PROMPT = (
    "Write a complete, playable 3D browser game in a SINGLE self-contained "
    "HTML file using three.js.\n\n"
    "Requirements:\n"
    "1. Load three.js from a CDN via an importmap and `import * as THREE from "
    "'three'` in a <script type=\"module\">. No build step, no bundler.\n"
    "2. The game is an endless lane runner: the player controls a cube that "
    "moves between 3 lanes, obstacles approach, and speed increases over time.\n"
    "3. Controls: arrow keys AND A/D. Also support touch or click on the left "
    "and right half of the screen.\n"
    "4. Real 3D: a perspective camera that follows the player, at least one "
    "directional light plus ambient, and visible ground with a sense of motion.\n"
    "5. Collision detection that ends the run.\n"
    "6. On-screen score that rises with distance, a visible game-over overlay "
    "showing the final score, and a restart that works without reloading the "
    "page.\n"
    "7. It must run correctly when the file is opened directly in a browser.\n\n"
    "Output ONLY one fenced ```html code block containing the entire file. No "
    "explanation before or after."
)

GAME_CHECKS = [
    ("html_doc", lambda s: "<html" in s.lower() and "</html>" in s.lower()),
    ("importmap", lambda s: "importmap" in s.lower()),
    ("three_import", lambda s: re.search(r"import\s+\*\s+as\s+THREE|from\s+['\"]three['\"]", s) is not None),
    ("renderer", lambda s: "WebGLRenderer" in s),
    ("perspective_cam", lambda s: "PerspectiveCamera" in s),
    ("light", lambda s: "Light" in s),
    # renderer.setAnimationLoop() is the idiom three.js actually recommends and
    # is required for WebXR, so checking only for requestAnimationFrame marks a
    # correct modern implementation as broken. Flash-Next's directive run used
    # it and was wrongly docked a point before this was widened.
    ("raf_loop", lambda s: "requestAnimationFrame" in s or "setAnimationLoop" in s),
    ("keyboard", lambda s: "keydown" in s.lower()),
    ("pointer", lambda s: any(k in s.lower() for k in ("touchstart", "pointerdown", "click"))),
    ("score", lambda s: "score" in s.lower()),
    ("restart", lambda s: any(k in s.lower() for k in ("restart", "replay", "play again", "tryagain", "try again"))),
]


def extract_html(text):
    m = re.findall(r"```(?:html)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return max(m, key=len).strip()
    i = text.lower().find("<!doctype")
    if i == -1:
        i = text.lower().find("<html")
    return text[i:].strip() if i != -1 else ""


def grade_game(response):
    """Structural score only. Returns (fraction, detail, html)."""
    html = extract_html(response)
    if not html:
        return 0.0, "no HTML block emitted", ""
    hits = [name for name, fn in GAME_CHECKS if fn(html)]
    misses = [name for name, fn in GAME_CHECKS if name not in hits]
    detail = "%d/%d structural" % (len(hits), len(GAME_CHECKS))
    if misses:
        detail += " missing=" + ",".join(misses)
    return len(hits) / len(GAME_CHECKS), detail, html

"""Self-test every grader in hardbench2.py before the real run.

A grader that accepts a wrong answer silently inflates a model's score, and a
grader that rejects a right answer silently deflates it. So each grader is
fed a known-good answer AND known-bad answers, including the specific wrong
answers a strong model is most likely to give (e.g. an O(n^2) inversion
count, a naive LRU, the tempting [(1, 2), (3, ...)] zip prediction).
"""

import io
import contextlib
import sys

sys.path.insert(0, "D:/Claude AI/AI Model Performance/"
                   "qwen3.8-vs-nemotron-vs-glimmer/research")

import hardbench2 as H  # noqa: E402

PASSED, FAILED = [], []


def check(name, got, want=True):
    if bool(got) == want:
        PASSED.append(name)
    else:
        FAILED.append(f"{name}: got {got}, wanted {want}")


# ------------------------------------------------ verify snippet output
SNIPPET = '''
def gen():
    yield 1
    yield 2
    yield 3

g = gen()
print(list(zip(g, g)))

d = {}
for k in ['a', 'b', 'a']:
    d[k] = d.get(k, 0) + 1
print(d)

x = [1, 2, 3]
y = x[:]
y.append(4)
print(x, y)

print([i for i in range(3)] == list(range(3)))
'''
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(SNIPPET, "<snippet>", "exec"), {})
REAL_OUTPUT = buf.getvalue()
print("ACTUAL interpreter output of the prediction task:")
print(REAL_OUTPUT)

check("output-prediction grader accepts real interpreter output",
      H.grade_output_prediction(REAL_OUTPUT))
# The most tempting wrong answer: treating zip(g, g) as pairing independently
check("output-prediction rejects [(1, 2), (3,)] style guess",
      H.grade_output_prediction(
          "[(1, 2), (3, 3)]\n{'a': 2, 'b': 1}\n[1, 2, 3] [1, 2, 3, 4]\nTrue"),
      False)
check("output-prediction rejects wrong dict",
      H.grade_output_prediction(
          "[(1, 2)]\n{'a': 1, 'b': 1}\n[1, 2, 3] [1, 2, 3, 4]\nTrue"), False)
check("output-prediction rejects aliased list answer",
      H.grade_output_prediction(
          "[(1, 2)]\n{'a': 2, 'b': 1}\n[1, 2, 3, 4] [1, 2, 3, 4]\nTrue"),
      False)

# ------------------------------------------------ numeric answer parsing
check("numeric accepts plain", H.answer_number_is("Answer: 1230", 1230))
check("numeric accepts commas", H.answer_number_is("Answer: 18,061", 18061))
check("numeric accepts fraction 9/4",
      H.answer_number_is("Answer: 9/4", 2.25, tol=1e-4))
check("numeric accepts decimal 2.25",
      H.answer_number_is("Answer: 2.25", 2.25, tol=1e-4))
check("numeric uses LAST Answer marker",
      H.answer_number_is("Answer: 100\nwait\nAnswer: 343", 343))
check("numeric rejects wrong value", H.answer_number_is("Answer: 342", 1230),
      False)
check("numeric rejects empty", H.answer_number_is("", 1230), False)

# ------------------------------------------------ instruction grader
GOOD_ACROSTIC = (
    "Peaks rise above the clouds every morning.\n"
    "Yesterday we climbed the steep northern ridge.\n"
    "Trails wind slowly through the quiet pines.\n"
    "Hikers carry water for the long ascent.\n"
    "Older guides know every hidden stone path.\n"
    "Night falls quickly across the cold summit.\n"
)
check("acrostic accepts valid 6x7", H.grade_instruction(GOOD_ACROSTIC))
check("acrostic rejects wrong word count",
      H.grade_instruction(GOOD_ACROSTIC.replace(
          "Peaks rise above the clouds every morning.",
          "Peaks rise above the clouds every single morning.")), False)
check("acrostic rejects wrong initial",
      H.grade_instruction(GOOD_ACROSTIC.replace(
          "Yesterday we climbed the steep northern ridge.",
          "Suddenly we climbed the steep northern ridge.")), False)
check("acrostic rejects 5 sentences",
      H.grade_instruction("\n".join(GOOD_ACROSTIC.strip().split("\n")[:5])),
      False)

# ------------------------------------------------ json grader
GOOD_JSON = """{
  "order_id": 7734, "date": "2026-07-19",
  "customer": {"name": "Ananya Iyer", "email": "ananya.i@example.com"},
  "items": [
    {"name": "Vertex Pro keyboard", "quantity": 3, "unit_price": 8750},
    {"name": "Nova mouse", "quantity": 2, "unit_price": 3200}
  ],
  "subtotal": 32650, "discount": 3265, "tax": 5289.30,
  "shipping_city": "Hyderabad", "total": 34674.30
}"""
check("json accepts correct", H.grade_json_hard(GOOD_JSON))
check("json rejects tax on pre-discount subtotal",
      H.grade_json_hard(GOOD_JSON.replace('"tax": 5289.30', '"tax": 5877.0')
                        .replace('"total": 34674.30', '"total": 35262.0')),
      False)
check("json rejects missing discount",
      H.grade_json_hard(GOOD_JSON.replace('"discount": 3265', '"discount": 0')),
      False)

# ------------------------------------------------ logic grid
check("logic grid accepts Chitra/Pune",
      "chitra" in H.final_answer("Answer: Chitra, Pune").lower()
      and "pune" in H.final_answer("Answer: Chitra, Pune").lower())

# ------------------------------------------------ executed code tests
GOOD_CALC = '''
def calc(expr):
    s = expr.replace(" ", "")
    pos = 0
    def peek():
        return s[pos] if pos < len(s) else ""
    def eat(c):
        nonlocal pos
        pos += len(c)
    def parse_expr():
        nonlocal pos
        v = parse_term()
        while peek() and peek() in "+-":
            op = peek(); eat(op)
            r = parse_term()
            v = v + r if op == "+" else v - r
        return v
    def parse_term():
        nonlocal pos
        v = parse_unary()
        while True:
            if s[pos:pos+2] == "//":
                eat("//"); v = v // parse_unary()
            elif peek() and peek() in "*/%" and s[pos:pos+2] != "**":
                op = peek(); eat(op); r = parse_unary()
                v = v * r if op == "*" else (v / r if op == "/" else v % r)
            else:
                break
        return v
    def parse_unary():
        nonlocal pos
        if peek() == "-":
            eat("-")
            return -parse_unary()
        return parse_power()
    def parse_power():
        nonlocal pos
        base = parse_atom()
        if s[pos:pos+2] == "**":
            eat("**")
            return base ** parse_unary()
        return base
    def parse_atom():
        nonlocal pos
        if peek() == "(":
            eat("("); v = parse_expr(); eat(")")
            return v
        start = pos
        while pos < len(s) and (s[pos].isdigit() or s[pos] == "."):
            pos += 1
        return float(s[start:pos]) if "." in s[start:pos] else int(s[start:pos])
    return parse_expr()
'''
CALC_FORBID = ["eval(", "exec(", "importast", "ast.", "compile("]
check("calc test accepts a correct evaluator",
      H.run_code_test(GOOD_CALC, H.CALC2_TEST, forbid=CALC_FORBID))
# eval() computes every case correctly, so only the static rule can reject it
check("calc REJECTS eval-based cheat via static rule",
      H.run_code_test("def calc(e):\n    return eval(e)\n",
                      H.CALC2_TEST, forbid=CALC_FORBID), False)
check("calc eval-cheat WOULD have passed without the static rule",
      H.run_code_test("def calc(e):\n    return eval(e)\n", H.CALC2_TEST))
# Realistic near-miss: correct everywhere except ** associativity
LEFT_ASSOC_CALC = GOOD_CALC.replace(
    """    def parse_power():
        nonlocal pos
        base = parse_atom()
        if s[pos:pos+2] == "**":
            eat("**")
            return base ** parse_unary()
        return base""",
    """    def parse_power():
        nonlocal pos
        base = parse_atom()
        while s[pos:pos+2] == "**":
            eat("**")
            base = base ** parse_atom()
        return base""",
)
assert LEFT_ASSOC_CALC != GOOD_CALC, "left-assoc patch did not apply"
check("calc REJECTS left-associative ** (2**3**2 = 64, not 512)",
      H.run_code_test(LEFT_ASSOC_CALC, H.CALC2_TEST, forbid=CALC_FORBID),
      False)

GOOD_INV = '''
def count_inversions(nums):
    def sort(a):
        if len(a) < 2:
            return a, 0
        m = len(a) // 2
        l, li = sort(a[:m]); r, ri = sort(a[m:])
        merged = []; i = j = 0; inv = li + ri
        while i < len(l) and j < len(r):
            if l[i] <= r[j]:
                merged.append(l[i]); i += 1
            else:
                merged.append(r[j]); j += 1; inv += len(l) - i
        merged += l[i:]; merged += r[j:]
        return merged, inv
    return sort(list(nums))[1]
'''
SLOW_INV = '''
def count_inversions(nums):
    n = len(nums); c = 0
    for i in range(n):
        for j in range(i+1, n):
            if nums[i] > nums[j]:
                c += 1
    return c
'''
check("inversions accepts O(n log n)",
      H.run_code_test(GOOD_INV, H.INVERSIONS_TEST))
check("inversions REJECTS O(n^2)",
      H.run_code_test(SLOW_INV, H.INVERSIONS_TEST, timeout=40), False)

GOOD_LRU = '''
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
'''
BAD_LRU = '''
class LRUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.d = {}
    def get(self, key):
        return self.d.get(key, -1)
    def put(self, key, value):
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.pop(next(iter(self.d)))
'''
check("LRU accepts O(1) OrderedDict version",
      H.run_code_test(GOOD_LRU, H.LRU_TEST))
check("LRU REJECTS no-recency-tracking version",
      H.run_code_test(BAD_LRU, H.LRU_TEST), False)

GOOD_CLOSURE = '''
def make_multipliers(factors):
    return [lambda x, f=f: x * f for f in factors]
'''
check("closure fix accepted", H.run_code_test(GOOD_CLOSURE,
                                              H.CLOSURE_FIX_TEST))
check("closure ORIGINAL buggy version rejected",
      H.run_code_test(
          "def make_multipliers(factors):\n"
          "    result = []\n"
          "    for f in factors:\n"
          "        result.append(lambda x: x * f)\n"
          "    return result\n", H.CLOSURE_FIX_TEST), False)

GOOD_VER = '''
def compare_versions(a, b):
    pa = [int(x) for x in a.split(".")]
    pb = [int(x) for x in b.split(".")]
    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa)); pb += [0] * (n - len(pb))
    for x, y in zip(pa, pb):
        if x != y:
            return 1 if x > y else -1
    return 0
'''
BAD_VER = '''
def compare_versions(a, b):
    return (a > b) - (a < b)
'''
check("version compare accepted", H.run_code_test(GOOD_VER, H.VERSION_TEST))
check("version compare REJECTS string comparison",
      H.run_code_test(BAD_VER, H.VERSION_TEST), False)

# ------------------------------------------------ needle prompt sanity
p = H.make_multihop_prompt()
words = len(p.split())
check("needle prompt is long", words > 35000)
check("needle contains all three facts",
      "Note A:" in p and "Note B:" in p and "Note C:" in p)
check("needle contains decoys", "Sunil" in p and "Priya" in p)
print(f"\nneedle prompt: {words:,} words, {len(p):,} chars")

# ------------------------------------------------ report
print("\n" + "=" * 62)
print(f"GRADER SELF-TEST: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 62)
for f in FAILED:
    print("  FAIL:", f)
if not FAILED:
    print("  All graders behave correctly on known-good and known-bad input.")

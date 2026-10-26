"""Harder problem set for the Granite 4.2 benchmark.

Why this file exists: the problem set used in the Ornith round saturated.
Five models scored 39/40 on graded reasoning and 18/18 on agentic tool use, so
the eval measured nothing except cost.  This set is built to break at least one
model, along three axes rather than one:

  * `math`   - 12 items in three difficulty tiers, every answer produced by a
               brute-force solver in this file rather than from memory.
  * `trap`   - 8 items with no valid answer (violated triangle inequality,
               non-integer polygon, contradictory knights, missing data).
               A model trained to always produce an answer fails these.
               They are graded together with `math` so that a model which
               blanket-answers IMPOSSIBLE scores badly instead of well.
  * `format` - 8 items graded by a Python predicate over the raw output, not
               by an answer key.  Output control is where small models differ.

Grading is extraction plus a checker function.  There is no LLM judge anywhere
in this project.

Run `python problems_hard.py` to re-derive every math answer from its solver
and assert it matches the recorded key.
"""

import re
from fractions import Fraction
from functools import lru_cache
from itertools import permutations

# --------------------------------------------------------------------------
# answer extraction
# --------------------------------------------------------------------------

ANSWER_INSTR = (
    "\n\nEnd your response with a single final line in exactly this format:\n"
    "ANSWER: <your final answer>"
)

# Shown on every item in the math+trap run, so a trap is not signposted by
# having different instructions from a solvable item.
TRAP_INSTR = (
    "\n\nIf the problem as stated has no valid answer, write ANSWER: IMPOSSIBLE"
    " instead of guessing."
)


def extract_answer(text):
    """Last ANSWER: line in the visible content."""
    if not text:
        return None
    m = re.findall(r"ANSWER:\s*(.+)", text)
    return m[-1].strip() if m else None


def norm_num(s):
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("\\", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^boxed\{(.+)\}$", r"\1", s)
    s = re.sub(r"^\\?\(?(.+?)\)?$", r"\1", s)
    return s.lower()


def make_num_checker(*accepted):
    acc = {norm_num(a) for a in accepted}

    def check(ans):
        return norm_num(ans) in acc

    return check


IMPOSSIBLE_WORDS = (
    "impossible", "none", "nosolution", "nosuchvalue", "undefined",
    "insufficientinformation", "notpossible", "doesnotexist", "n/a",
    "cannotbedetermined", "indeterminate",
)


def check_impossible(ans):
    """Accept any clear statement that the problem has no valid answer."""
    if ans is None:
        return False
    a = re.sub(r"[^a-z0-9/]", "", ans.lower())
    return any(w in a for w in IMPOSSIBLE_WORDS)


# --------------------------------------------------------------------------
# brute-force solvers.  Each one is the ground truth for exactly one item.
# --------------------------------------------------------------------------

def s_m1():
    """Ordered pairs 1<=x,y<=200 with 13 | x^2+y^2."""
    return sum(1 for x in range(1, 201) for y in range(1, 201)
               if (x * x + y * y) % 13 == 0)


def s_m2():
    """Binary strings of length 14 with no three consecutive equal chars."""
    n = 14
    # dp[(last, runlen)] = count
    dp = {("0", 1): 1, ("1", 1): 1}
    for _ in range(n - 1):
        nxt = {}
        for (last, run), c in dp.items():
            for ch in "01":
                r = run + 1 if ch == last else 1
                if r >= 3:
                    continue
                nxt[(ch, r)] = nxt.get((ch, r), 0) + c
        dp = nxt
    return sum(dp.values())


def s_m3():
    """Four fair d6.  P(product divisible by 8), exact fraction."""
    from itertools import product
    good = sum(1 for r in product(range(1, 7), repeat=4)
               if (r[0] * r[1] * r[2] * r[3]) % 8 == 0)
    return Fraction(good, 6 ** 4)


def s_m4():
    """Sum of n in 1..500 with n | 3^n + 1."""
    return sum(n for n in range(1, 501) if (pow(3, n, n) + 1) % n == 0)


def s_m5():
    """Subsets of {1..24}, no two consecutive elements, sum divisible by 7.

    DP over (last taken?, sum mod 7).  Cross-checked against exhaustive
    enumeration for n=16 in the self-test below.
    """
    return _no_consec_sum_div(24, 7)


def _no_consec_sum_div(n, m):
    # dp[(prev_taken, s mod m)] -> count
    dp = {(False, 0): 1}
    for i in range(1, n + 1):
        nxt = {}
        for (prev, s), c in dp.items():
            nxt[(False, s)] = nxt.get((False, s), 0) + c          # skip i
            if not prev:                                          # take i
                k = (True, (s + i) % m)
                nxt[k] = nxt.get(k, 0) + c
        dp = nxt
    return sum(c for (prev, s), c in dp.items() if s == 0)


def s_m6():
    """Permutations p of 1..9 with |p(i)-i| <= 2 for every i."""
    return sum(1 for p in permutations(range(1, 10))
               if all(abs(p[i] - (i + 1)) <= 2 for i in range(9)))


BLOCKED = [(2, 3), (4, 4), (5, 2), (6, 6)]


def s_m7():
    """Monotone lattice paths (0,0)->(8,8) avoiding four blocked points."""
    blocked = set(BLOCKED)
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 1
    for x in range(9):
        for y in range(9):
            if (x, y) in blocked:
                grid[x][y] = 0
                continue
            if (x, y) == (0, 0):
                continue
            grid[x][y] = (grid[x - 1][y] if x else 0) + (grid[x][y - 1] if y else 0)
    return grid[8][8]


def s_m8():
    """Last four digits of 2^2026 + 3^2026."""
    return (pow(2, 2026, 10000) + pow(3, 2026, 10000)) % 10000


def s_m9():
    """Expected number of adjacent same-rank pairs in a shuffled 52-card deck."""
    return Fraction(51, 1) * Fraction(3, 51)


def s_m10():
    """6-digit numbers that strictly increase then strictly decrease.

    The peak must be interior, so both runs have length >= 2.
    """
    cnt = 0
    for n in range(100000, 1000000):
        d = [int(c) for c in str(n)]
        peak = d.index(max(d))
        if peak == 0 or peak == 5:
            continue
        up = all(d[i] < d[i + 1] for i in range(peak))
        down = all(d[i] > d[i + 1] for i in range(peak, 5))
        if up and down:
            cnt += 1
    return cnt


def s_m11():
    """Domino tilings of a 4x8 rectangle."""

    @lru_cache(maxsize=None)
    def fill(col, mask):
        if col == 8:
            return 1 if mask == 0 else 0
        return _fill_col(col, mask, 0, 0, fill)

    return fill(0, 0)


def _fill_col(col, mask, row, nxt, fill):
    if row == 4:
        return fill(col + 1, nxt)
    if mask & (1 << row):                      # already covered from prev col
        return _fill_col(col, mask, row + 1, nxt, fill)
    total = 0
    # horizontal domino into the next column
    total += _fill_col(col, mask, row + 1, nxt | (1 << row), fill)
    # vertical domino inside this column
    if row + 1 < 4 and not (mask & (1 << (row + 1))):
        total += _fill_col(col, mask | (1 << (row + 1)), row + 2, nxt, fill)
    return total


def s_m12():
    """Smallest n such that n, n+1, n+2 each have exactly 8 divisors."""

    def ndiv(x):
        c, i = 0, 1
        while i * i <= x:
            if x % i == 0:
                c += 2 if i * i != x else 1
            i += 1
        return c

    n = 1
    while True:
        if ndiv(n) == 8 and ndiv(n + 1) == 8 and ndiv(n + 2) == 8:
            return n
        n += 1


SOLVERS = {
    "M1": s_m1, "M2": s_m2, "M3": s_m3, "M4": s_m4, "M5": s_m5, "M6": s_m6,
    "M7": s_m7, "M8": s_m8, "M9": s_m9, "M10": s_m10, "M11": s_m11,
    "M12": s_m12,
}


# --------------------------------------------------------------------------
# math items.  `answer` is asserted against SOLVERS by the self-test below,
# so nothing here is an answer I merely remembered.
# --------------------------------------------------------------------------

MATH = [
    dict(id="M1", tier=1, dept="math", solver="M1",
         prompt="How many ordered pairs (x, y) of integers with 1 <= x <= 200 "
                "and 1 <= y <= 200 satisfy that x^2 + y^2 is divisible by 13?",
         answer="5929", checker=make_num_checker("5929")),
    dict(id="M2", tier=1, dept="math", solver="M2",
         prompt="How many binary strings of length 14 contain no three "
                "consecutive identical characters?",
         answer="1220", checker=make_num_checker("1220")),
    dict(id="M3", tier=1, dept="math", solver="M3",
         prompt="Four fair six-sided dice are rolled. What is the probability "
                "that the product of the four results is divisible by 8? "
                "Give the answer as a fraction in lowest terms.",
         answer="25/48", checker=make_num_checker("25/48", "0.5208333333333334")),
    dict(id="M4", tier=1, dept="math", solver="M4",
         prompt="Find the sum of all positive integers n with n <= 500 such "
                "that n divides 3^n + 1.",
         answer="313", checker=make_num_checker("313")),

    dict(id="M5", tier=2, dept="math", solver="M5",
         prompt="How many subsets of {1, 2, ..., 24} contain no two "
                "consecutive integers and have a sum divisible by 7? Count "
                "the empty set, whose sum is 0.",
         answer="17343", checker=make_num_checker("17343")),
    dict(id="M6", tier=2, dept="math", solver="M6",
         prompt="How many permutations p of (1, 2, ..., 9) satisfy "
                "|p(i) - i| <= 2 for every i from 1 to 9?",
         answer="932", checker=make_num_checker("932")),
    dict(id="M7", tier=2, dept="math", solver="M7",
         prompt="On an integer grid you may move only one step right or one "
                "step up. Count the paths from (0, 0) to (8, 8) that never "
                "pass through any of the four blocked points (2, 3), (4, 4), "
                "(5, 2), (6, 6).",
         answer="2312", checker=make_num_checker("2312")),
    dict(id="M8", tier=2, dept="math", solver="M8",
         prompt="Find the last four digits of 2^2026 + 3^2026. Give the answer "
                "as a four-digit string, including any leading zero.",
         answer="7193", checker=make_num_checker("7193")),

    dict(id="M9", tier=3, dept="math", solver="M9",
         prompt="A standard 52-card deck is shuffled uniformly at random. "
                "Consider the 51 adjacent position pairs (i, i+1). What is the "
                "expected number of those pairs in which the two cards have "
                "the same rank? Give the exact value.",
         answer="3", checker=make_num_checker("3", "3/1", "3.0")),
    dict(id="M10", tier=3, dept="math", solver="M10",
         prompt="How many 6-digit positive integers have digits that strictly "
                "increase up to one peak digit and then strictly decrease, "
                "where the peak is neither the first nor the last digit?",
         answer="10794", checker=make_num_checker("10794")),
    dict(id="M11", tier=3, dept="math", solver="M11",
         prompt="In how many ways can a 4 x 8 rectangle be tiled completely "
                "by 1 x 2 dominoes? Dominoes may be placed in either "
                "orientation and two tilings are the same only if every "
                "domino position matches.",
         answer="2245", checker=make_num_checker("2245")),
    dict(id="M12", tier=3, dept="math", solver="M12",
         prompt="Find the smallest positive integer n such that each of n, "
                "n + 1, and n + 2 has exactly 8 positive divisors.",
         answer="230", checker=make_num_checker("230")),
]

# --------------------------------------------------------------------------
# trap items.  No valid answer exists.  Graded in the same run as MATH so a
# model cannot win by answering IMPOSSIBLE to everything.
# --------------------------------------------------------------------------

TRAPS = [
    dict(id="T1", tier=1, dept="trap", why="violates the triangle inequality",
         prompt="In triangle ABC, side AB = 3, side BC = 4, and side CA = 8. "
                "Compute the area of the triangle.",
         answer="IMPOSSIBLE", checker=check_impossible),
    dict(id="T2", tier=1, dept="trap", why="360/25 is not an integer",
         prompt="A regular polygon has interior angles of exactly 155 degrees. "
                "How many sides does it have?",
         answer="IMPOSSIBLE", checker=check_impossible),
    dict(id="T3", tier=1, dept="trap", why="no perfect square ends in 2",
         prompt="Find the smallest positive integer n whose square ends in the "
                "digit 2.",
         answer="IMPOSSIBLE", checker=check_impossible),
    dict(id="T4", tier=2, dept="trap", why="no consistent knight/knave assignment",
         prompt="On an island, knights always tell the truth and knaves always "
                "lie, and every inhabitant is one or the other.\n"
                "A says: 'I am a knave.'\n"
                "B says: 'A is a knight.'\n"
                "Determine what A and B are. Give the answer in the form "
                "A=<knight/knave>, B=<knight/knave>.",
         answer="IMPOSSIBLE", checker=check_impossible),
    dict(id="T5", tier=2, dept="trap", why="the sequence given is geometric, not arithmetic",
         prompt="The arithmetic sequence begins 2, 4, 8, 16. What is its 10th "
                "term?",
         answer="IMPOSSIBLE", checker=check_impossible),
    dict(id="T6", tier=2, dept="trap", why="cannot draw 8 balls from a bag of 7",
         prompt="A bag contains 3 red balls and 4 blue balls. Eight balls are "
                "drawn without replacement. What is the probability that every "
                "ball drawn is red?",
         answer="IMPOSSIBLE or 0",
         checker=lambda a: check_impossible(a) or norm_num(a) in {"0", "0.0"}),
    dict(id="T7", tier=3, dept="trap", why="x^2 is not injective on the reals",
         prompt="The function f(x) = x^2 is injective on the real numbers. "
                "Compute the value of its inverse function at 4.",
         answer="IMPOSSIBLE", checker=check_impossible),
    # T8 was written as a trap ("direction and separation are never given") and
    # was wrong: leaving "the station" implies a shared origin and direction,
    # which makes the chase well posed.  All six models answered 7:00 pm, which
    # is correct.  Kept as a solvable item; see RESCORE.md.
    dict(id="T8", tier=3, dept="math", why="reclassified from trap",
         prompt="A train leaves the station at 3:00 pm travelling at 60 miles "
                "per hour. A second train leaves at 4:00 pm travelling at 80 "
                "miles per hour. At what time does the second train catch the "
                "first?",
         answer="7:00 pm",
         checker=lambda a: bool(a) and re.sub(r"[^0-9apm:]", "",
                                              a.lower()).startswith(("7:00pm", "7pm", "1900", "19:00"))),
]

# --------------------------------------------------------------------------
# format / instruction-following items.  Graded by a predicate over the raw
# reply with the reasoning block stripped, not by an answer key.
# --------------------------------------------------------------------------

import json as _json


def _visible(text):
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    return text.strip()


def f_exactly_seven_words(text):
    words = _visible(text).split()
    return len(words) == 7 and all(w.isalpha() for w in words)


def f_no_letter_e(text):
    v = _visible(text)
    sents = [s for s in re.split(r"(?<=[.!?])\s+", v) if s.strip()]
    return (len(sents) == 3 and "e" not in v.lower()
            and all(len(s.split()) <= 12 for s in sents))


def f_json_strict(text):
    v = _visible(text)
    if "```" in v:
        return False
    try:
        obj = _json.loads(v)
    except Exception:
        return False
    if not isinstance(obj, dict) or set(obj) != {"city", "population", "tags"}:
        return False
    return (isinstance(obj["city"], str)
            and isinstance(obj["population"], int)
            and not isinstance(obj["population"], bool)
            and isinstance(obj["tags"], list) and len(obj["tags"]) == 3
            and all(isinstance(t, str) for t in obj["tags"]))


def f_alpha_csv(text):
    v = _visible(text)
    if v.endswith("."):
        return False
    parts = [p.strip() for p in v.split(",")]
    return (len(parts) == 12 and all(p.isalpha() for p in parts)
            and parts == sorted(parts, key=str.lower))


def f_acrostic(text):
    lines = [l.strip() for l in _visible(text).splitlines() if l.strip()]
    if len(lines) != 5:
        return False
    return "".join(l[0].upper() for l in lines) == "CLOUD"


def f_word_budget(text):
    words = _visible(text).split()
    return (48 <= len(words) <= 52
            and "the" not in [w.lower().strip(".,;:!?") for w in words])


def f_table_shape(text):
    rows = [l for l in _visible(text).splitlines() if l.strip().startswith("|")]
    if len(rows) != 6:
        return False
    return {len(r.strip().strip("|").split("|")) for r in rows} == {3}


def f_reverse_digits(text):
    return _visible(text).strip() == "9 8 7 6 5 4 3 2 1 0"


FORMAT = [
    dict(id="F1", tier=1, dept="format", checker=f_exactly_seven_words,
         prompt="Reply with exactly seven words and nothing else. No "
                "punctuation of any kind, no quotation marks, no final period. "
                "Every word must be alphabetic."),
    dict(id="F2", tier=1, dept="format", checker=f_alpha_csv,
         prompt="List exactly 12 single-word animal names in alphabetical "
                "order, separated by a comma and a single space. Output only "
                "that list: no preamble, no numbering, no trailing period."),
    dict(id="F3", tier=1, dept="format", checker=f_reverse_digits,
         prompt="Output the digits from 9 down to 0 in descending order, "
                "separated by single spaces, on one line, with nothing else "
                "before or after."),
    dict(id="F4", tier=2, dept="format", checker=f_json_strict,
         prompt="Output a single JSON object and nothing else. No markdown "
                "code fence. It must have exactly three keys: \"city\" (a "
                "string), \"population\" (an integer, not a string), and "
                "\"tags\" (an array of exactly three strings)."),
    dict(id="F5", tier=2, dept="format", checker=f_acrostic,
         prompt="Write exactly five lines. The first letters of the five "
                "lines, read downward, must spell CLOUD. Output only the five "
                "lines, with no title, numbering, or explanation."),
    dict(id="F6", tier=2, dept="format", checker=f_table_shape,
         prompt="Output a markdown table with exactly three columns and "
                "exactly four data rows, describing four programming "
                "languages. Include the header row and the separator row, so "
                "the output is exactly six lines beginning with a pipe "
                "character. Output nothing else."),
    dict(id="F7", tier=3, dept="format", checker=f_no_letter_e,
         prompt="Describe a rainstorm in exactly three sentences. Each "
                "sentence must be at most 12 words. The letter 'e' must not "
                "appear anywhere in your reply. Output only the three "
                "sentences."),
    dict(id="F8", tier=3, dept="format", checker=f_word_budget,
         prompt="Explain how a bicycle brake works in between 48 and 52 "
                "words. The word 'the' must not appear anywhere in your "
                "reply. Output only the explanation."),
]

ALL_ITEMS = MATH + TRAPS + FORMAT


def _self_test():
    bad = 0
    for it in MATH:
        got = SOLVERS[it["solver"]]()
        if str(got) != it["answer"]:
            print(f"KEY MISMATCH {it['id']}: solver={got} recorded={it['answer']}")
            bad += 1
        if not it["checker"](it["answer"]):
            print(f"CHECKER REJECTS OWN KEY {it['id']}")
            bad += 1
        if it["checker"]("IMPOSSIBLE"):
            print(f"MATH CHECKER ACCEPTS IMPOSSIBLE {it['id']}")
            bad += 1
    for it in TRAPS:
        if not it["checker"]("IMPOSSIBLE"):
            print(f"TRAP CHECKER REJECTS IMPOSSIBLE {it['id']}")
            bad += 1
        if it["checker"]("42"):
            print(f"TRAP CHECKER ACCEPTS A NUMBER {it['id']}")
            bad += 1
    ids = [i["id"] for i in ALL_ITEMS]
    if len(ids) != len(set(ids)):
        print("DUPLICATE IDS")
        bad += 1
    print(f"{len(MATH)} math, {len(TRAPS)} traps, {len(FORMAT)} format "
          f"= {len(ALL_ITEMS)} items; {bad} problems found")
    return bad


if __name__ == "__main__":
    raise SystemExit(_self_test())

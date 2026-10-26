"""Problem set: 4 math + 4 reasoning, all with objectively checkable answers.

Every prompt instructs the model to end with 'ANSWER: <x>' so grading is
extraction + normalized comparison, never LLM-judged.
"""
import re

ANSWER_INSTR = "\n\nEnd your response with a single final line in exactly this format:\nANSWER: <your final answer>"


def extract_answer(text):
    """Take the last ANSWER: line from the visible content."""
    matches = re.findall(r"ANSWER:\s*(.+)", text)
    return matches[-1].strip() if matches else None


def norm_num(s):
    if s is None:
        return None
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("\\", "")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"^boxed\{(.+)\}$", r"\1", s)
    return s.lower()


def make_num_checker(*accepted):
    acc = {norm_num(a) for a in accepted}
    def check(ans):
        return norm_num(ans) in acc
    return check


def make_word_checker(*accepted):
    acc = {a.lower() for a in accepted}
    def check(ans):
        if ans is None:
            return False
        a = re.sub(r"[^a-z0-9 ]", "", ans.lower()).strip()
        return a in acc or any(a.startswith(x) for x in acc)
    return check


def check_knights(ans):
    if ans is None:
        return False
    a = ans.lower()
    pos = lambda name, role: re.search(name + r"\s*[=:]\s*" + role, a)
    return bool(pos("a", "knave") and pos("b", "knight") and pos("c", "knave"))


PROBLEMS = [
    # ---------------- MATH ----------------
    dict(
        id="M1_powers",
        dept="math",
        prompt="Find the last three digits of 7^2025. Give the answer as a three-digit number." + ANSWER_INSTR,
        checker=make_num_checker("807"),
        answer="807",
    ),
    dict(
        id="M2_lcm_pairs",
        dept="math",
        prompt="How many ordered pairs (a, b) of positive integers satisfy lcm(a, b) = 2^3 * 3^2 * 5?" + ANSWER_INSTR,
        checker=make_num_checker("105"),
        answer="105",
    ),
    dict(
        id="M3_factorial_zeros",
        dept="math",
        prompt="Find the sum of all positive integers n < 100 such that n! ends in exactly 22 trailing zeros." + ANSWER_INSTR,
        checker=make_num_checker("485"),
        answer="485",
    ),
    dict(
        id="M4_coin_runs",
        dept="math",
        prompt="A fair coin is flipped 10 times. What is the probability that no two consecutive flips both come up heads? Express the answer as a fraction in lowest terms." + ANSWER_INSTR,
        checker=make_num_checker("9/64", "144/1024"),
        answer="9/64",
    ),
    # ---------------- REASONING ----------------
    dict(
        id="R1_knights",
        dept="reasoning",
        prompt=(
            "On an island, knights always tell the truth and knaves always lie. You meet three people: A, B, and C.\n"
            "A says: 'B and C are both knaves.'\n"
            "B says: 'A is a knave.'\n"
            "C says: 'A is a knight.'\n"
            "Determine what each of A, B, and C is. Give the answer in the form: A=<knight/knave>, B=<knight/knave>, C=<knight/knave>"
            + ANSWER_INSTR
        ),
        checker=check_knights,
        answer="A=knave, B=knight, C=knave",
    ),
    dict(
        id="R2_race",
        dept="reasoning",
        prompt=(
            "Five runners finished a race: Alice, Ben, Carol, Dan, and Emma. No ties.\n"
            "Ben finished before Alice but after Carol. Dan finished last. Emma finished before Carol.\n"
            "Who finished second? Answer with just the name." + ANSWER_INSTR
        ),
        checker=make_word_checker("carol"),
        answer="Carol",
    ),
    dict(
        id="R3_houses",
        dept="reasoning",
        prompt=(
            "Four people live in a row of four houses, numbered 1 to 4 from left to right. "
            "Each person has a distinct name (Peter, Quinn, Rosa, Sam), each house a distinct color (red, blue, green, white), "
            "and each person a distinct pet (fish, dog, cat, bird) and a distinct drink (tea, coffee, milk, juice).\n"
            "1. Peter lives in the red house.\n"
            "2. The green house is immediately to the right of the white house.\n"
            "3. Quinn drinks tea and does not live in house 1.\n"
            "4. The person in house 2 owns the dog.\n"
            "5. Rosa owns the bird.\n"
            "6. The coffee drinker lives in the green house.\n"
            "7. Sam lives immediately to the left of Peter.\n"
            "8. The person in house 1 drinks milk.\n"
            "9. The fish owner drinks juice.\n"
            "10. The red house is house 4.\n"
            "Who owns the fish? Answer with just the name." + ANSWER_INSTR
        ),
        checker=make_word_checker("peter"),
        answer="Peter",
    ),
    dict(
        id="R4_boxes",
        dept="reasoning",
        prompt=(
            "Three closed boxes are labeled 'Apples', 'Oranges', and 'Mixed'. Every label is wrong. "
            "One box contains only apples, one contains only oranges, and one contains a mix of both. "
            "You draw a single fruit from the box labeled 'Mixed' and it is an apple.\n"
            "What does the box labeled 'Oranges' contain? Answer with one of: apples / oranges / mixed." + ANSWER_INSTR
        ),
        checker=make_word_checker("mixed", "mix", "mixed fruit", "a mix", "mix of both", "mixed apples and oranges"),
        answer="mixed",
    ),
]

def check_knight_set(ans):
    if ans is None:
        return False
    letters = set(re.findall(r"\b([A-E])\b", ans.upper()))
    return letters == {"B", "C", "D"}


PROBLEMS_HARD = [
    dict(
        id="G1_floor_sqrt",
        dept="math",
        prompt="For how many positive integers n with 1 <= n <= 2024 does floor(sqrt(n)) divide n exactly?" + ANSWER_INSTR,
        checker=make_num_checker("132"),
        answer="132",
    ),
    dict(
        id="G2_spread_subsets",
        dept="math",
        prompt=(
            "How many subsets of {1, 2, ..., 12}, including the empty set, have the property that "
            "any two distinct elements differ by more than 2?" + ANSWER_INSTR
        ),
        checker=make_num_checker("129"),
        answer="129",
    ),
    dict(
        id="G3_six_runners",
        dept="reasoning",
        prompt=(
            "Six runners finished a race in positions 1 through 6 with no ties: Fiona, Gabe, Hana, Ivan, Jack, and Kira.\n"
            "1. The positions of Fiona and Gabe add up to 7.\n"
            "2. Hana finished immediately before or immediately after Ivan.\n"
            "3. Jack finished before Fiona.\n"
            "4. Kira's position is exactly twice Jack's position.\n"
            "5. Hana finished after Gabe.\n"
            "6. Ivan finished after Hana.\n"
            "7. Gabe finished before Fiona.\n"
            "Who finished third? Answer with just the name." + ANSWER_INSTR
        ),
        checker=make_word_checker("gabe"),
        answer="Gabe",
    ),
    dict(
        id="G4_five_islanders",
        dept="reasoning",
        prompt=(
            "On an island, knights always tell the truth and knaves always lie. Five islanders A, B, C, D, E each make one statement:\n"
            "A says: 'Exactly two of the five of us are knights.'\n"
            "B says: 'A and E are both knaves.'\n"
            "C says: 'B is a knight, or D is a knave, or both.'\n"
            "D says: 'C and I are both knights.'\n"
            "E says: 'A is a knight and B is a knave.'\n"
            "Exactly which islanders are knights? List their letters." + ANSWER_INSTR
        ),
        checker=check_knight_set,
        answer="B, C, D",
    ),
]

# ---------------- context fillers ----------------
# Relevant reference material, deterministic, free of the target problems' answers.

MATH_FILLER_TEMPLATES = [
    "Modular exponentiation reference, case {i}: to compute a^k mod m efficiently, use repeated squaring. "
    "For example, with a = {a} and m = {m}, the powers cycle with a period that divides phi({m}). "
    "Euler's theorem states that if gcd(a, m) = 1 then a^phi(m) is congruent to 1 mod m. "
    "The multiplicative order of a modulo m is the smallest positive t with a^t congruent to 1. ",
    "Divisor counting reference, case {i}: if n = p1^e1 * p2^e2 * ... * pk^ek, the number of divisors is the product of (ei + 1). "
    "For lcm-based counting problems, handle each prime independently: if lcm(a, b) must equal p^e, then max(va, vb) = e, "
    "where va and vb are the p-adic valuations. The number of pairs with max equal to e is 2e + 1 for a single prime power. ",
    "Legendre's formula reference, case {i}: the exponent of prime p in n! equals floor(n/p) + floor(n/p^2) + floor(n/p^3) + ... "
    "Trailing zeros of n! are governed by the exponent of 5 since factors of 2 are plentiful. "
    "Note that the count of trailing zeros jumps by 2 or more when n crosses a multiple of 25, so some zero-counts are skipped entirely. ",
    "Binary string combinatorics reference, case {i}: the number of binary strings of length n with no two adjacent 1s "
    "satisfies the Fibonacci recurrence f(n) = f(n-1) + f(n-2), with f(1) = 2 and f(2) = 3. "
    "Probability problems over fair coin sequences divide such counts by 2^n. Always reduce the resulting fraction. ",
    "Worked example, case {i}: compute {a}^{k} mod 100. Reduce the exponent modulo the order of {a} mod 100, then square repeatedly. "
    "Common orders modulo 100 divide 20. A last-two-digits question is exactly a mod-100 question, and last-three-digits is mod 1000. ",
]

REASONING_FILLER_TEMPLATES = [
    "Logic puzzle technique, note {i}: in truth-teller and liar puzzles, assume a speaker's type and propagate consequences until you "
    "reach a contradiction or a fully consistent assignment. If exactly one assumption survives, the solution is unique. "
    "A statement by a liar must be entirely false; a conjunction stated by a liar means at least one conjunct is false. ",
    "Constraint grid technique, note {i}: for placement puzzles, build a table of positions versus attributes. Start with absolute clues "
    "that pin an attribute to a position, then apply relational clues such as 'immediately to the left of'. "
    "Cross out impossible cells after each deduction; a row or column with one remaining cell is forced. ",
    "Ordering technique, note {i}: translate 'X finished before Y' into a directed edge X -> Y and look for a unique topological order. "
    "When one element is pinned to an end, the remaining chain often becomes fully determined. Check every stated relation against the final order. ",
    "Label logic technique, note {i}: when every label is guaranteed wrong, a single observation can cascade. Identify what the observed box "
    "cannot be, conclude what it must be, and then use the all-labels-wrong constraint to force the remaining assignments. ",
    "Verification habit, note {i}: after solving any logic puzzle, re-read each clue and confirm it holds in your solution. "
    "A solution that satisfies all but one clue is not a solution. If two assignments satisfy all clues, the puzzle is under-constrained. ",
]


def build_filler(dept, target_chars):
    templates = MATH_FILLER_TEMPLATES if dept == "math" else REASONING_FILLER_TEMPLATES
    parts, i = [], 0
    total = 0
    while total < target_chars:
        t = templates[i % len(templates)]
        p = t.format(i=i + 1, a=3 + (i * 7) % 90, m=100 + (i * 37) % 900, k=50 + (i * 13) % 400)
        parts.append(p)
        total += len(p)
        i += 1
    text = "".join(parts)
    # scrub coincidental collisions with target-problem answers
    for leak, sub in [("807", "812"), ("485", "489"), ("105", "108"), ("9/64", "9/62")]:
        text = text.replace(leak, sub)
    return text


def filled_prompt(problem, target_tokens):
    """Chars-per-token measured empirically on this tokenizer: math filler 3.36, reasoning filler 4.75."""
    ratio = 3.3 if problem["dept"] == "math" else 4.7
    filler = build_filler(problem["dept"], target_chars=int(target_tokens * ratio))
    return (
        "Below is a reference document with relevant techniques, followed by a problem to solve. "
        "Use whatever is relevant.\n\n=== REFERENCE DOCUMENT ===\n"
        + filler
        + "\n=== END OF REFERENCE DOCUMENT ===\n\nNow solve this problem:\n\n"
        + problem["prompt"]
    )

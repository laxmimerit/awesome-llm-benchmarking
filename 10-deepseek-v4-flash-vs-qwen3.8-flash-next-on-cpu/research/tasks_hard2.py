"""Tier 2: problems a model that scores 12/12 on tier 1 should still lose points on.

Tier 1 saturates. glimmer (a 30B) scored 12/12 on it in the earlier run, so it
cannot separate two frontier-size models. This tier uses three mechanisms that
resist saturation:

  math      unavoidable casework, no memorized identity shortcut
  algorithm a correct-but-naive solution passes correctness and fails wall clock
  debugging a function that is right on obvious inputs and wrong on one edge

Every math ground truth here was computed by brute force in
research/gt_hard2.py, never recalled. The numbers are 70, 1456 and 3600.

The algorithm tasks are the sharpest discriminator because they cannot be
bluffed: the subprocess test either finishes inside the limit or it does not.
"""
import re

from tasks import answer_number_is, run_code_test  # noqa: F401

# ------------------------------------------------------------------ code tests

RANGE_SUMS_TEST = """
import random, time

def ref(nums, lower, upper):
    pre = [0]
    for x in nums:
        pre.append(pre[-1] + x)
    c = 0
    for i in range(len(nums)):
        for j in range(i + 1, len(pre)):
            if lower <= pre[j] - pre[i] <= upper:
                c += 1
    return c

random.seed(11)
for n in [0, 1, 5, 40, 200]:
    a = [random.randint(-30, 30) for _ in range(n)]
    lo, hi = -12, 37
    assert count_range_sums(a, lo, hi) == ref(a, lo, hi), (n, a[:8])

big = [random.randint(-10**5, 10**5) for _ in range(100000)]
t0 = time.time()
got = count_range_sums(big, -5000, 5000)
elapsed = time.time() - t0
assert isinstance(got, int) and got >= 0
assert elapsed < 6.0, f"too slow: {elapsed:.1f}s (needs O(n log n))"
print("PASS")
"""

MANACHER_TEST = """
import random, time

def ref(s):
    if not s:
        return ""
    best = s[0]
    for i in range(len(s)):
        for d in (0, 1):
            l, r = i, i + d
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            if r - l - 1 > len(best):
                best = s[l + 1:r]
    return best

random.seed(5)
for n in [0, 1, 2, 30, 300]:
    s = "".join(random.choice("ab") for _ in range(n))
    got = longest_palindrome(s)
    want = ref(s)
    assert len(got) == len(want), (n, got, want)
    assert got in s and got == got[::-1], (n, got)

# Timing input must be ADVERSARIAL, not random. In a random binary string the
# longest palindrome is only ~2*log2(n) chars, so expand-around-centre exits
# almost immediately at every centre and an O(n^2) solution finishes in 0.5s.
# "abab...ab" forces deep expansion from most centres, which is the real
# quadratic worst case. Its longest palindrome has length len(big)-1.
big = "ab" * 100000
t0 = time.time()
got = longest_palindrome(big)
elapsed = time.time() - t0
assert got in big and got == got[::-1]
assert len(got) == len(big) - 1, f"expected length {len(big) - 1}, got {len(got)}"
assert elapsed < 4.0, f"too slow: {elapsed:.1f}s (needs O(n))"
print("PASS")
"""

PRECISION_TEST = """
# The reference is exact integer arithmetic; the candidate must match it
# including at magnitudes where float sqrt/pow drift by one.
def ref(n, k):
    lo, hi = 0, 1
    while hi ** k <= n:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= n:
            lo = mid
        else:
            hi = mid - 1
    return lo

cases = [(0, 2), (1, 2), (8, 3), (9, 2), (15, 2), (16, 2),
         (10**18, 2), (10**18, 3), (10**18 - 1, 3),
         (999999999999999999, 2), (2**60, 6), (2**60 - 1, 6),
         (10**30, 5), (10**30 - 1, 5), (7**20, 20), (7**20 - 1, 20)]
for n, k in cases:
    got = integer_root(n, k)
    want = ref(n, k)
    assert got == want, f"integer_root({n}, {k}) = {got}, want {want}"
print("PASS")
"""

TASKS_HARD2 = [
    {
        "id": "h2_lcm_triples",
        "cat": "math",
        "prompt": ("Find the number of ordered triples (a, b, c) of positive "
                   "integers satisfying all three conditions: lcm(a, b) = 1000, "
                   "lcm(b, c) = 2000, and lcm(c, a) = 2000. Show your reasoning, "
                   "then end your reply with 'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 70),
    },
    {
        "id": "h2_subsets_nonconsecutive",
        "cat": "math",
        "prompt": ("How many 5-element subsets of {1, 2, ..., 20} contain no two "
                   "consecutive integers AND have an element sum divisible by 3? "
                   "Show your reasoning, then end your reply with "
                   "'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 1456),
    },
    {
        "id": "h2_digits_distinct_sum",
        "cat": "math",
        "prompt": ("How many 6-digit positive integers have six pairwise "
                   "distinct digits that sum to exactly 20? Leading zeros are "
                   "not allowed. Show your reasoning, then end your reply with "
                   "'Answer: <number>'."),
        "check": lambda r: answer_number_is(r, 3600),
    },
    {
        "id": "h2_code_range_sums",
        "cat": "code",
        "prompt": ("Write a Python function "
                   "`count_range_sums(nums: list[int], lower: int, upper: int) "
                   "-> int` returning how many contiguous subarrays have a sum "
                   "in the inclusive range [lower, upper]. It must handle "
                   "n = 100,000 with values up to 1e5 in magnitude in under 6 "
                   "seconds, so an O(n^2) scan will not pass. Reply with only "
                   "the code in one fenced python block."),
        "check": ("code", RANGE_SUMS_TEST),
    },
    {
        "id": "h2_code_manacher",
        "cat": "code",
        "prompt": ("Write a Python function "
                   "`longest_palindrome(s: str) -> str` returning the longest "
                   "palindromic substring of s. It must handle a 200,000 "
                   "character string in under 4 seconds, so the O(n^2) "
                   "expand-around-centre approach will not pass. Return any one "
                   "of the longest if there are ties, and return '' for empty "
                   "input. Reply with only the code in one fenced python block."),
        "check": ("code", MANACHER_TEST),
    },
    {
        "id": "h2_code_integer_root",
        "cat": "code",
        "prompt": ("Write a Python function `integer_root(n: int, k: int) -> int` "
                   "returning floor(n ** (1/k)) exactly, for n >= 0 and k >= 1, "
                   "with n as large as 10**30. It must be exact: any approach "
                   "using floating point sqrt, pow, or ** with a fractional "
                   "exponent will drift by one at large magnitudes and fail. "
                   "Reply with only the code in one fenced python block."),
        "check": ("code", PRECISION_TEST),
    },
]

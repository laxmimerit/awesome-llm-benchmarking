"""Compute every ground-truth answer used by the hard benchmark.

Nothing in the benchmark is hand-derived. Each value below is produced by
exact computation or exhaustive enumeration, printed here so the expected
answers baked into hardbench2.py can be checked against a real run.
"""

from fractions import Fraction
from functools import lru_cache
from itertools import permutations

print("=" * 62)
print("GROUND TRUTH VERIFICATION")
print("=" * 62)

# ---- M1: sum of repunits 1 + 11 + 111 + ... (300 terms), digit sum
N = sum(int("1" * k) for k in range(1, 301))
m1 = sum(int(c) for c in str(N))
print(f"M1 repunit-300 digit sum      = {m1}   (N has {len(str(N))} digits)")

# ---- M2: last three digits of 7^(7^7)
m2 = pow(7, 7 ** 7, 1000)
print(f"M2 7^(7^7) mod 1000           = {m2}")

# ---- M3: expected rolls of a fair d6 until three consecutive sixes
# States 0,1,2 = current run length of sixes. E_i = 1 + (1/6)E_{i+1} + (5/6)E_0
# Solve exactly with Fractions via elimination.
# E0 = 1 + (1/6)E1 + (5/6)E0 ; E1 = 1 + (1/6)E2 + (5/6)E0 ; E2 = 1 + (5/6)E0
E0 = Fraction(0)
# substitute: solve linear system directly
# E2 = 1 + (5/6)E0
# E1 = 1 + (1/6)(1 + (5/6)E0) + (5/6)E0
# E0 = 1 + (1/6)E1 + (5/6)E0  -> (1/6)E0 = 1 + (1/6)E1
from sympy import Rational, solve, symbols  # noqa: E402

e0, e1, e2 = symbols("e0 e1 e2")
sol = solve(
    [
        e0 - (1 + Rational(1, 6) * e1 + Rational(5, 6) * e0),
        e1 - (1 + Rational(1, 6) * e2 + Rational(5, 6) * e0),
        e2 - (1 + Rational(5, 6) * e0),
    ],
    [e0, e1, e2],
    dict=True,
)[0]
m3 = sol[e0]
print(f"M3 E[rolls to three 6s]       = {m3}")

# ---- M4: domino tilings of a 4 x 10 board (broken-profile DP)
def tilings(rows, cols):
    @lru_cache(maxsize=None)
    def fill(col, mask, nmask):
        # mask: cells already filled in current column; nmask: next column
        if col == cols:
            return 1 if mask == 0 and nmask == 0 else 0
        full = (1 << rows) - 1
        if mask == full:
            return fill(col + 1, nmask, 0)
        r = 0
        while mask >> r & 1:
            r += 1
        total = 0
        # vertical domino
        if r + 1 < rows and not (mask >> (r + 1) & 1):
            total += fill(col, mask | (1 << r) | (1 << (r + 1)), nmask)
        # horizontal domino
        if col + 1 < cols and not (nmask >> r & 1):
            total += fill(col, mask | (1 << r), nmask | (1 << r))
        return total

    return fill(0, 0, 0)


m4 = tilings(4, 10)
print(f"M4 4x10 domino tilings        = {m4}   (sanity 2x12={tilings(2, 12)})")

# ---- M5: arrangements of MISSISSIPPI with no two S adjacent
word = "MISSISSIPPI"
seen = set()
for p in permutations(word):
    seen.add("".join(p))
m5_total = len(seen)
m5 = sum(1 for s in seen if "SS" not in s)
print(f"M5 MISSISSIPPI no two S adj   = {m5}   (of {m5_total} distinct)")

# ---- M6: expected draws until first blue, bag of 5 red + 3 blue
bag = ["R"] * 5 + ["B"] * 3
tot = Fraction(0)
cnt = 0
for p in set(permutations(bag)):
    draws = p.index("B") + 1
    tot += draws
    cnt += 1
m6 = tot / cnt
print(f"M6 E[draws to first blue]     = {m6} = {float(m6)}")

# ---- R1: logic grid, brute-force uniqueness check
people = ["Aditi", "Bhavesh", "Chitra", "Dev", "Esha"]
cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Pune"]
pets = ["cat", "dog", "parrot", "fish", "rabbit"]

sols = []
for cperm in permutations(cities):
    C = dict(zip(people, cperm))
    for pperm in permutations(pets):
        P = dict(zip(people, pperm))
        owner = {v: k for k, v in P.items()}
        liver = {v: k for k, v in C.items()}
        # 1. The dog owner lives in Mumbai.
        if C[owner["dog"]] != "Mumbai":
            continue
        # 2. Chitra lives in Chennai.
        if C["Chitra"] != "Chennai":
            continue
        # 3. Bhavesh owns the parrot.
        if P["Bhavesh"] != "parrot":
            continue
        # 4. The cat owner lives in Delhi.
        if C[owner["cat"]] != "Delhi":
            continue
        # 5. Aditi does not live in Delhi or Mumbai.
        if C["Aditi"] in ("Delhi", "Mumbai"):
            continue
        # 6. Dev does not own the dog.
        if P["Dev"] == "dog":
            continue
        # 7. The person in Pune owns the rabbit.
        if P[liver["Pune"]] != "rabbit":
            continue
        # 8. Esha does not live in Kolkata.
        if C["Esha"] == "Kolkata":
            continue
        sols.append((C, P))

print(f"R1 logic grid solutions       = {len(sols)}")
for C, P in sols:
    print("   ", {k: (C[k], P[k]) for k in people})

# ---- R4: three-hop needle arithmetic
locker = 214
code = locker * 3 + 7          # 649
final = code + 128             # shelf id
print(f"R4 three-hop final answer     = {final}  (214*3+7={code}, +128)")

print("=" * 62)
print("SUMMARY OF EXPECTED ANSWERS")
print("=" * 62)
for k, v in [
    ("m1_repunit_digit_sum", m1),
    ("m2_power_tower_mod", m2),
    ("m3_expected_three_sixes", m3),
    ("m4_tilings_4x10", m4),
    ("m5_mississippi_no_ss", m5),
    ("m6_expected_first_blue", f"{m6} ({float(m6)})"),
    ("r4_three_hop", final),
]:
    print(f"  {k:28} {v}")

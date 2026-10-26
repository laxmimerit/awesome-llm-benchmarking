"""Design + brute-force verify the hard tier (pass G)."""
from itertools import permutations, product

# G1: count n <= 2024 with floor(sqrt(n)) | n
import math
cnt = sum(1 for n in range(1, 2025) if n % math.isqrt(n) == 0)
print("G1 answer:", cnt)
assert cnt == 132

# G2: subsets of {1..12} with no two elements differing by 1 or 2
total = 0
for mask in range(1 << 12):
    els = [i + 1 for i in range(12) if mask >> i & 1]
    if all(b - a > 2 for a, b in zip(els, els[1:])):
        total += 1
print("G2 answer:", total)

# G3: 6 runners row puzzle -> designed below, verify uniqueness
# People: F G H I J K positions 1..6
sols = []
for perm in permutations("FGHIJK"):
    p = {c: i + 1 for i, c in enumerate(perm)}
    if p["F"] + p["G"] != 7: continue            # F and G positions sum to 7
    if abs(p["H"] - p["I"]) != 1: continue       # H adjacent to I
    if p["J"] >= p["F"]: continue                # J before F
    if p["K"] != 2 * p["J"]: continue            # K position twice J's
    if p["H"] <= p["G"]: continue                # H after G
    if p["I"] <= p["H"]: continue                # I after H
    if p["G"] >= p["F"]: continue                # G before F
    sols.append(perm)
print("G3 solutions:", sols)

# G4: five islanders, count-based statements
best = None
for st in range(1):
    sols4 = []
    for bits in product([0, 1], repeat=5):
        A, B, C, D, E = bits
        k = sum(bits)
        sA = (k == 2)
        sB = (A == 0 and E == 0)
        sC = (B == 1 or D == 0)
        sD = (C == 1 and D == 1)
        sE = (A == 1 and B == 0)
        ok = (sA == A) and (sB == B) and (sC == C) and (sD == D) and (sE == E)
        if ok:
            sols4.append(bits)
    print("G4 solutions:", sols4)

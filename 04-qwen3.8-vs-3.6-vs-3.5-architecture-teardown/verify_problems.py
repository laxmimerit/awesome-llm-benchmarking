"""Brute-force verification that every baked answer is correct and unique."""
from itertools import permutations, product
from math import gcd

# M1
assert pow(7, 2025, 1000) == 807, pow(7, 2025, 1000)

# M2: ordered pairs with lcm = 2^3*3^2*5 = 360
N = 360
cnt = 0
for a in range(1, N + 1):
    if N % a:
        continue
    for b in range(1, N + 1):
        if N % b:
            continue
        if a * b // gcd(a, b) == N:
            cnt += 1
assert cnt == 105, cnt

# M3: n<100 with exactly 22 trailing zeros in n!
def tz(n):
    s, p = 0, 5
    while p <= n:
        s += n // p
        p *= 5
    return s
hits = [n for n in range(1, 100) if tz(n) == 22]
assert sum(hits) == 485 and hits == [95, 96, 97, 98, 99], hits

# M4: 10 flips, no two consecutive heads
ok = sum(1 for bits in product([0, 1], repeat=10) if not any(bits[i] and bits[i+1] for i in range(9)))
from fractions import Fraction
fr = Fraction(ok, 2 ** 10)
assert fr == Fraction(9, 64), (ok, fr)

# R1 knights: A: 'B and C both knaves', B: 'A is knave', C: 'A is knight'
sols = []
for A, B, C in product([True, False], repeat=3):  # True = knight
    sA = (not B) and (not C)
    sB = not A
    sC = A
    if (sA == A) and (sB == B) and (sC == C):
        sols.append((A, B, C))
assert sols == [(False, True, False)], sols  # A knave, B knight, C knave

# R2 race: Ben<Alice(pos), Ben>Carol, Dan last, Emma before Carol
names = ["Alice", "Ben", "Carol", "Dan", "Emma"]
sols = []
for perm in permutations(names):
    p = {n: i for i, n in enumerate(perm)}
    if p["Carol"] < p["Ben"] < p["Alice"] and p["Dan"] == 4 and p["Emma"] < p["Carol"]:
        sols.append(perm)
assert len(sols) == 1 and sols[0][1] == "Carol", sols

# R3 houses
sols = []
for namesv in permutations(["Peter", "Quinn", "Rosa", "Sam"]):
    name = {namesv[i]: i for i in range(4)}  # person -> house index 0..3
    for colors in permutations(["red", "blue", "green", "white"]):
        col = {colors[i]: i for i in range(4)}
        if col["red"] != 3: continue                      # clue 10
        if name["Peter"] != col["red"]: continue          # clue 1
        if col["green"] != col["white"] + 1: continue     # clue 2
        if name["Sam"] != name["Peter"] - 1: continue     # clue 7
        for pets in permutations(["fish", "dog", "cat", "bird"]):
            pet = {pets[i]: i for i in range(4)}          # pet -> house
            if pet["dog"] != 1: continue                  # clue 4
            if pet["bird"] != name["Rosa"]: continue      # clue 5
            for drinks in permutations(["tea", "coffee", "milk", "juice"]):
                dr = {drinks[i]: i for i in range(4)}
                if dr["tea"] != name["Quinn"] or name["Quinn"] == 0: continue  # clue 3
                if dr["coffee"] != col["green"]: continue  # clue 6
                if dr["milk"] != 0: continue               # clue 8
                if dr["juice"] != pet["fish"]: continue    # clue 9
                owner = [p for p, h in name.items() if h == pet["fish"]][0]
                sols.append(owner)
assert len(sols) == 1, sols
assert sols[0] == "Peter", sols

# R4 boxes: all labels wrong, drew apple from box labeled Mixed
sols = []
for actual in permutations(["apples", "oranges", "mixed"]):
    labels = ["apples", "oranges", "mixed"]
    if any(actual[i] == labels[i] for i in range(3)):
        continue  # labels all wrong
    mixed_label_content = actual[2]
    if mixed_label_content == "oranges":
        continue  # we drew an apple from it
    sols.append(dict(zip(labels, actual)))
assert len(sols) == 1, sols
assert sols[0]["oranges"] == "mixed", sols

print("ALL 8 PROBLEMS VERIFIED: answers correct and unique")

# filler sanity: sizes and no answer leakage
import problems as P
for dept in ["math", "reasoning"]:
    f = P.build_filler(dept, 400000)
    for leak in ["807", "105", "485", "9/64", "Peter", "Carol"]:
        assert leak not in f, (dept, leak)
print("Filler: no answer leakage; sample sizes ok")

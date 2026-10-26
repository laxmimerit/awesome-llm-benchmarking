"""Executable coding tasks, graded by running hidden tests.

Design note.  Standard interview problems (two-sum, valid parentheses, islands)
saturate immediately on modern 8B-and-up models, which is exactly the failure
the Ornith round ran into.  Every task here is instead a *precise specification*
with edge cases that are stated but easy to skip: strict Roman numerals, INI
line continuations, deterministic tie-breaking, negative business-day offsets.
The reference implementation lives next to the task, and the self-test asserts
the reference passes every hidden test, so a failing model is failing the spec
rather than failing my test harness.

Model output is executed locally in a separate short-lived process.  That is
deliberate and is the only way to grade code objectively, but it is worth
saying out loud: this file runs code written by the model under test.
"""

import re

CODE_INSTR = (
    "\n\nRespond with a single Python 3 code block and nothing else. Define "
    "exactly the requested function at module level. Do not include tests, "
    "examples, or explanation after the code block. Do not read stdin."
)


def extract_code(text):
    """Pull the last fenced Python block, falling back to the whole reply."""
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[-1]
    return text


# --------------------------------------------------------------------------
# reference implementations
# --------------------------------------------------------------------------

def _ref_roman(s):
    if not isinstance(s, str) or not s:
        return None
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    if any(c not in vals for c in s):
        return None
    total, i = 0, 0
    pairs = {"IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900}
    while i < len(s):
        if i + 1 < len(s) and s[i:i + 2] in pairs:
            total += pairs[s[i:i + 2]]
            i += 2
        else:
            total += vals[s[i]]
            i += 1
    if total > 3999:
        return None
    return total if _ref_to_roman(total) == s else None


def _ref_to_roman(n):
    table = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
             (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
             (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, sym in table:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


def _ref_expand(spec):
    out = []
    if not spec.strip():
        return out
    for part in spec.split(","):
        part = part.strip()
        if "-" in part[1:]:
            i = part.index("-", 1)
            a, b = int(part[:i]), int(part[i + 1:])
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(part))
    return out


def _ref_compress(nums):
    if not nums:
        return ""
    runs, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        runs.append((start, prev))
        start = prev = n
    runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def _ref_interval_subtract(a, b):
    out = []
    for lo, hi in a:
        pieces = [(lo, hi)]
        for blo, bhi in b:
            nxt = []
            for plo, phi in pieces:
                if bhi < plo or blo > phi:
                    nxt.append((plo, phi))
                    continue
                if plo < blo:
                    nxt.append((plo, blo - 1))
                if phi > bhi:
                    nxt.append((bhi + 1, phi))
            pieces = nxt
        out.extend(pieces)
    return sorted(p for p in out if p[0] <= p[1])


def _ref_topo(deps):
    import heapq
    nodes = set(deps)
    for vs in deps.values():
        nodes |= set(vs)
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for node, needs in deps.items():
        for need in needs:
            adj[need].append(node)
            indeg[node] += 1
    heap = sorted(n for n in nodes if indeg[n] == 0)
    heapq.heapify(heap)
    out = []
    while heap:
        n = heapq.heappop(heap)
        out.append(n)
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                heapq.heappush(heap, m)
    if len(out) != len(nodes):
        raise ValueError("cycle")
    return out


def _ref_parse_ini(text):
    data, section = {}, ""
    pending = None
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if pending is not None:
            line = pending + line.strip()
            pending = None
        stripped = line.strip()
        if not stripped or stripped[0] in ";#":
            continue
        if stripped.endswith("\\"):
            pending = stripped[:-1]
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            data.setdefault(section, {})
            continue
        if "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        k, v = k.strip(), v.strip()
        sec = data.setdefault(section, {})
        if k in sec:
            if isinstance(sec[k], list):
                sec[k].append(v)
            else:
                sec[k] = [sec[k], v]
        else:
            sec[k] = v
    return data


def _ref_csv_split(line):
    out, cur, i, inq = [], [], 0, False
    while i < len(line):
        c = line[i]
        if inq:
            if c == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 2
                    continue
                inq = False
                i += 1
                continue
            cur.append(c)
            i += 1
        else:
            if c == '"' and not cur:
                inq = True
                i += 1
            elif c == ",":
                out.append("".join(cur))
                cur = []
                i += 1
            else:
                cur.append(c)
                i += 1
    out.append("".join(cur))
    return out


def _ref_business(start, n, holidays):
    import datetime as dt
    d = dt.date.fromisoformat(start)
    hol = set(holidays)
    step = 1 if n >= 0 else -1
    left = abs(n)
    while left:
        d += dt.timedelta(days=step)
        if d.weekday() < 5 and d.isoformat() not in hol:
            left -= 1
    return d.isoformat()


def _ref_wrap(text, width):
    out = []
    for para in text.split("\n"):
        words, line = para.split(), ""
        for w in words:
            if not line:
                line = w
            elif len(line) + 1 + len(w) <= width:
                line += " " + w
            else:
                out.append(line)
                line = w
        out.append(line)
    return out


REFERENCES = {
    "C1": _ref_roman,
    "C2": _ref_expand,
    "C3": _ref_compress,
    "C4": _ref_interval_subtract,
    "C5": _ref_topo,
    "C6": _ref_parse_ini,
    "C7": _ref_csv_split,
    "C8": _ref_business,
    "C9": _ref_wrap,
}


# --------------------------------------------------------------------------
# tasks.  `tests` is executed after the model's code in a fresh process.
# --------------------------------------------------------------------------

TASKS = [
    dict(
        id="C1", tier=1, func="roman_to_int_strict",
        prompt=(
            "Write a Python function `roman_to_int_strict(s)` that converts a "
            "Roman numeral string to an int.\n\n"
            "Rules:\n"
            "- Return the integer value only if `s` is the single canonical "
            "Roman numeral for a value from 1 to 3999. Canonical means it is "
            "exactly the string the standard conversion algorithm produces, so "
            "'IIII', 'VIIII', 'IC' and 'MMMM' are all invalid.\n"
            "- Return None for anything non-canonical, for an empty string, "
            "for lowercase input, for a non-string argument, and for any "
            "string containing a character outside IVXLCDM.\n"
            "- Do not raise."
        ),
        tests="""
assert roman_to_int_strict("IV") == 4
assert roman_to_int_strict("MCMXCIV") == 1994
assert roman_to_int_strict("MMMCMXCIX") == 3999
assert roman_to_int_strict("I") == 1
assert roman_to_int_strict("IIII") is None
assert roman_to_int_strict("VIIII") is None
assert roman_to_int_strict("IC") is None
assert roman_to_int_strict("MMMM") is None
assert roman_to_int_strict("") is None
assert roman_to_int_strict("iv") is None
assert roman_to_int_strict("XYZ") is None
assert roman_to_int_strict(None) is None
assert roman_to_int_strict(4) is None
assert roman_to_int_strict("XL") == 40
assert roman_to_int_strict("LXXXIX") == 89
""",
    ),
    dict(
        id="C2", tier=1, func="expand_ranges",
        prompt=(
            "Write a Python function `expand_ranges(spec)` that expands a "
            "compact range specification into a list of ints.\n\n"
            "Rules:\n"
            "- `spec` is a comma-separated list of parts. A part is either a "
            "single integer or two integers joined by a hyphen.\n"
            "- Integers may be negative, so the hyphen that separates a range "
            "is never the first character of the part. For example '-3--1' "
            "expands to [-3, -2, -1].\n"
            "- A descending range is allowed and counts down: '9-7' expands to "
            "[9, 8, 7].\n"
            "- Parts may be surrounded by spaces.\n"
            "- An empty or whitespace-only spec returns [].\n"
            "- Output preserves the order the parts are written in and does "
            "not deduplicate."
        ),
        tests="""
assert expand_ranges("1-3,5,7-9") == [1,2,3,5,7,8,9]
assert expand_ranges("") == []
assert expand_ranges("   ") == []
assert expand_ranges("4") == [4]
assert expand_ranges("-5") == [-5]
assert expand_ranges("-3--1") == [-3,-2,-1]
assert expand_ranges("9-7") == [9,8,7]
assert expand_ranges(" 1 - 2 , 4 ") == [1,2,4]
assert expand_ranges("3-3") == [3]
assert expand_ranges("2,2,1-2") == [2,2,1,2]
assert expand_ranges("-1-1") == [-1,0,1]
""",
    ),
    dict(
        id="C3", tier=1, func="compress_ranges",
        prompt=(
            "Write a Python function `compress_ranges(nums)` that compresses a "
            "list of ints into a compact string.\n\n"
            "Rules:\n"
            "- Scan the list in the order given and group only maximal runs "
            "where each element is exactly one more than the previous "
            "element.\n"
            "- A run of length 1 is written as the number itself; a longer run "
            "is written as 'first-last'.\n"
            "- Join the groups with a single comma and no spaces.\n"
            "- An empty list returns the empty string.\n"
            "- Do not sort and do not deduplicate: a descending pair such as "
            "[3, 2] is two separate groups."
        ),
        tests="""
assert compress_ranges([1,2,3,5,7,8,9]) == "1-3,5,7-9"
assert compress_ranges([]) == ""
assert compress_ranges([4]) == "4"
assert compress_ranges([3,2]) == "3,2"
assert compress_ranges([-3,-2,-1]) == "-3--1"
assert compress_ranges([1,1,2]) == "1,1-2"
assert compress_ranges([5,6]) == "5-6"
assert compress_ranges([0,1,2,10]) == "0-2,10"
assert compress_ranges([7,7,7]) == "7,7,7"
""",
    ),
    dict(
        id="C4", tier=2, func="interval_subtract",
        prompt=(
            "Write a Python function `interval_subtract(a, b)` operating on "
            "closed integer intervals given as (low, high) tuples with low <= "
            "high.\n\n"
            "Rules:\n"
            "- `a` and `b` are lists of such intervals. Neither list is "
            "necessarily sorted and the intervals in `b` may overlap each "
            "other.\n"
            "- Return the set difference: every integer that is in some "
            "interval of `a` and in no interval of `b`.\n"
            "- Express the result as a list of maximal closed intervals, "
            "sorted ascending, with no empty intervals.\n"
            "- Because these are integer intervals, subtracting (5, 5) from "
            "(1, 10) gives [(1, 4), (6, 10)].\n"
            "- Do not merge intervals that came from different entries of `a` "
            "unless they were already adjacent in the result."
        ),
        tests="""
assert interval_subtract([(1,10)], [(5,5)]) == [(1,4),(6,10)]
assert interval_subtract([(1,10)], []) == [(1,10)]
assert interval_subtract([], [(1,2)]) == []
assert interval_subtract([(1,5)], [(1,5)]) == []
assert interval_subtract([(1,5)], [(0,10)]) == []
assert interval_subtract([(1,5)], [(6,9)]) == [(1,5)]
assert interval_subtract([(1,20)], [(5,8),(7,12)]) == [(1,4),(13,20)]
assert interval_subtract([(10,20),(1,5)], [(3,12)]) == [(1,2),(13,20)]
assert interval_subtract([(1,3)], [(2,2),(3,3)]) == [(1,1)]
assert interval_subtract([(-5,5)], [(-1,1)]) == [(-5,-2),(2,5)]
assert interval_subtract([(1,1)], [(1,1)]) == []
""",
    ),
    dict(
        id="C5", tier=2, func="topo_order",
        prompt=(
            "Write a Python function `topo_order(deps)` returning a "
            "topological ordering.\n\n"
            "Rules:\n"
            "- `deps` is a dict mapping a node name to a list of names that "
            "must come before it.\n"
            "- A name that appears only inside a dependency list is still a "
            "node.\n"
            "- Among all valid orderings return the lexicographically smallest "
            "one, comparing the output sequences as lists of strings. The "
            "standard way to get this is to always take the alphabetically "
            "smallest currently available node.\n"
            "- Raise ValueError if no ordering exists.\n"
            "- An empty dict returns []."
        ),
        tests="""
assert topo_order({}) == []
assert topo_order({"a": []}) == ["a"]
assert topo_order({"b": ["a"]}) == ["a","b"]
assert topo_order({"z": [], "a": []}) == ["a","z"]
assert topo_order({"c": ["a"], "b": []}) == ["a","b","c"]
assert topo_order({"shirt": [], "tie": ["shirt"], "jacket": ["tie","shirt"]}) == ["shirt","tie","jacket"]
assert topo_order({"d": ["b","c"], "c": ["a"], "b": ["a"]}) == ["a","b","c","d"]
try:
    topo_order({"a": ["b"], "b": ["a"]})
    raise AssertionError("expected ValueError")
except ValueError:
    pass
try:
    topo_order({"a": ["a"]})
    raise AssertionError("expected ValueError")
except ValueError:
    pass
""",
    ),
    dict(
        id="C6", tier=2, func="parse_ini",
        prompt=(
            "Write a Python function `parse_ini(text)` that parses a small INI "
            "dialect into a dict of dicts.\n\n"
            "Rules:\n"
            "- A section header is a line whose stripped form starts with '[' "
            "and ends with ']'. The section name is the text between the "
            "brackets, stripped.\n"
            "- Keys before any section header belong to the section named by "
            "the empty string ''.\n"
            "- A key line is 'key = value'; split on the first '=' only, and "
            "strip both sides.\n"
            "- A line whose stripped form is empty, or starts with ';' or "
            "'#', is ignored.\n"
            "- If a line's stripped form ends with a backslash, drop the "
            "backslash and concatenate it with the stripped next line, "
            "inserting no separator, before parsing the joined result.\n"
            "- If the same key appears twice in one section, the value becomes "
            "a list of the values in order of appearance; three occurrences "
            "give a list of three.\n"
            "- A section header with no keys still produces an empty dict for "
            "that section.\n"
            "- Lines that are neither comments, headers, nor contain '=' are "
            "ignored."
        ),
        tests="""
assert parse_ini("") == {}
assert parse_ini("a = 1") == {"": {"a": "1"}}
assert parse_ini("[s]\\nk=v") == {"s": {"k": "v"}}
assert parse_ini("[s]") == {"s": {}}
assert parse_ini("; c\\n# d\\n[s]\\nk=v") == {"s": {"k": "v"}}
assert parse_ini("[s]\\nk=1\\nk=2") == {"s": {"k": ["1","2"]}}
assert parse_ini("[s]\\nk=1\\nk=2\\nk=3") == {"s": {"k": ["1","2","3"]}}
assert parse_ini("[ s ]\\nk = v ") == {"s": {"k": "v"}}
assert parse_ini("[s]\\nk = a=b") == {"s": {"k": "a=b"}}
assert parse_ini("[s]\\nk = one\\\\\\n  two") == {"s": {"k": "onetwo"}}
assert parse_ini("[s]\\ngarbage line\\nk=v") == {"s": {"k": "v"}}
assert parse_ini("a=1\\n[s]\\nb=2") == {"": {"a": "1"}, "s": {"b": "2"}}
""",
    ),
    dict(
        id="C7", tier=3, func="csv_split",
        prompt=(
            "Write a Python function `csv_split(line)` that splits one CSV "
            "record into a list of field strings.\n\n"
            "Rules:\n"
            "- Fields are separated by commas.\n"
            "- A double quote starts a quoted field only when it is the first "
            "character of that field; inside a quoted field, two consecutive "
            "double quotes mean one literal double quote, and the closing "
            "quote ends the quoting.\n"
            "- A double quote that appears after other characters in an "
            "unquoted field is a literal character.\n"
            "- Whitespace is never stripped.\n"
            "- An empty line returns [''].\n"
            "- A trailing comma produces a final empty field.\n"
            "- Do not use the csv module."
        ),
        tests="""
assert csv_split("") == [""]
assert csv_split("a,b,c") == ["a","b","c"]
assert csv_split("a,,c") == ["a","","c"]
assert csv_split("a,b,") == ["a","b",""]
assert csv_split('"a,b",c') == ["a,b","c"]
assert csv_split('"say ""hi""' + '",x') == ['say "hi"',"x"]
assert csv_split('a"b,c') == ['a"b',"c"]
assert csv_split(' a , b ') == [" a "," b "]
assert csv_split('"",x') == ["","x"]
assert csv_split('"a"') == ["a"]
assert csv_split('x,"y,z",w') == ["x","y,z","w"]
""",
    ),
    dict(
        id="C8", tier=3, func="add_business_days",
        prompt=(
            "Write a Python function `add_business_days(start, n, holidays)` "
            "that moves a date by a number of business days.\n\n"
            "Rules:\n"
            "- `start` is an ISO date string 'YYYY-MM-DD'. `holidays` is a "
            "list of ISO date strings.\n"
            "- Business days are Monday to Friday and not in `holidays`.\n"
            "- Move forward for positive `n` and backward for negative `n`, "
            "counting only business days landed on. `n == 0` returns `start` "
            "unchanged even if `start` is a weekend or holiday.\n"
            "- The starting day is never counted, only the days moved to.\n"
            "- Return an ISO date string.\n"
            "- The standard library is available."
        ),
        tests="""
assert add_business_days("2026-08-26", 0, []) == "2026-08-26"
assert add_business_days("2026-08-26", 1, []) == "2026-08-27"
assert add_business_days("2026-08-28", 1, []) == "2026-08-31"
assert add_business_days("2026-08-31", -1, []) == "2026-08-28"
assert add_business_days("2026-08-26", 5, []) == "2026-09-02"
assert add_business_days("2026-08-26", 1, ["2026-08-27"]) == "2026-08-28"
assert add_business_days("2026-08-29", 1, []) == "2026-08-31"
assert add_business_days("2026-08-29", -1, []) == "2026-08-28"
assert add_business_days("2026-08-26", -5, []) == "2026-08-19"
assert add_business_days("2026-08-26", 10, ["2026-08-31","2026-09-01"]) == "2026-09-11"
assert add_business_days("2026-01-01", 0, ["2026-01-01"]) == "2026-01-01"
""",
    ),
    dict(
        id="C9", tier=3, func="wrap_lines",
        prompt=(
            "Write a Python function `wrap_lines(text, width)` that greedily "
            "wraps text and returns a list of lines.\n\n"
            "Rules:\n"
            "- Split `text` on newline characters into paragraphs and wrap "
            "each paragraph independently, concatenating the results.\n"
            "- Within a paragraph, split on runs of whitespace, then greedily "
            "put as many words on a line as fit, where a line fits if its "
            "length is at most `width`. Words on a line are joined by exactly "
            "one space.\n"
            "- A word longer than `width` goes on its own line and is never "
            "split.\n"
            "- An empty paragraph produces one empty string in the output, so "
            "wrap_lines('', 5) returns [''].\n"
            "- Never emit trailing spaces."
        ),
        tests="""
assert wrap_lines("", 5) == [""]
assert wrap_lines("a b c", 5) == ["a b c"]
assert wrap_lines("a b c", 3) == ["a b","c"]
assert wrap_lines("aaaaaa", 3) == ["aaaaaa"]
assert wrap_lines("aa bbbbbb cc", 3) == ["aa","bbbbbb","cc"]
assert wrap_lines("one\\ntwo", 10) == ["one","two"]
assert wrap_lines("one\\n\\ntwo", 10) == ["one","","two"]
assert wrap_lines("a   b", 5) == ["a b"]
assert wrap_lines("the quick brown fox", 10) == ["the quick","brown fox"]
assert wrap_lines("  padded  ", 20) == ["padded"]
""",
    ),
]


def _self_test():
    """Every reference implementation must pass its own hidden tests."""
    bad = 0
    for t in TASKS:
        ns = {t["func"]: REFERENCES[t["id"]]}
        try:
            exec(compile(t["tests"], f"<tests {t['id']}>", "exec"), ns)
            print(f"  ref {t['id']} OK")
        except Exception as e:
            print(f"  ref {t['id']} FAILED: {type(e).__name__}: {e}")
            bad += 1
    print(f"{len(TASKS)} coding tasks; {bad} reference failures")
    return bad


if __name__ == "__main__":
    raise SystemExit(_self_test())

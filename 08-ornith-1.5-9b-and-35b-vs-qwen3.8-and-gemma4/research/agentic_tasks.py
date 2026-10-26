"""Tasks for the agentic eval, with objective graders.

Each task is solvable only by chaining tool calls: the answer is not present in
the prompt and cannot be guessed from general knowledge. Graders normalise
whitespace and case but never accept a near miss.
"""

import re

from agentic_env import DB_TOOLS, REPO_TOOLS, REPO_FIX_LINE

ANSWER_INSTR = (
    "\n\nWhen you have the answer, reply with a final line in exactly this "
    "format and nothing after it:\nANSWER: <your answer>"
)


def extract(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return m[-1].strip() if m else None


def _norm(s):
    return re.sub(r"[^a-z0-9+]+", " ", (s or "").lower()).strip()


def contains_all(*needles):
    def check(ans):
        a = _norm(ans)
        return all(_norm(n) in a for n in needles)
    return check


def equals_any(*accepted):
    acc = {_norm(a) for a in accepted}
    def check(ans):
        return _norm(ans) in acc
    return check


TASKS = [
    # --- db: 3 hops (employee -> region -> incidents) ---------------------
    {
        "id": "A1_hop3_datacenter",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 3,
        # DEFECTIVE AS ORIGINALLY WORDED, kept here with the fix applied.
        # The first wording ended "...and report the timezone of that region",
        # which reads as asking for the timezone alone. All five models
        # answered exactly "UTC+1" and all five were marked wrong by a grader
        # demanding both fields. That is an ambiguous prompt, not a capability
        # difference, so the item is excluded from the reported scores of the
        # run that used the old wording.
        "prompt": (
            "Using the tools, find which datacenter the employee named "
            "'Cai Mwangi' works out of. Report BOTH the datacenter and the "
            "timezone of that region, separated by a comma."
        ),
        # Cai -> region_id 1 -> Nordwest / dc-alpha / UTC+1
        "checker": contains_all("dc-alpha", "UTC+1"),
        "expected": "dc-alpha, UTC+1",
        "defective_in_run": "2026-08-25 agentic.jsonl",
    },
    {
        "id": "A2_hop4_manager_region",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 3,
        "prompt": (
            "Using the tools, find the manager of the employee named "
            "'Jo Lindqvist', then report the NAME of the region that manager "
            "works in."
        ),
        # Jo -> manager 4 (Dara Olsen) -> region 2 -> Sudost
        "checker": equals_any("Sudost", "Sudost region", "the Sudost region"),
        "expected": "Sudost",
    },
    {
        "id": "A3_aggregate_incidents",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 3,
        "prompt": (
            "Using the tools, find the datacenter for the region that "
            "'Fenna Brill' works in, then report the total incident minutes "
            "recorded at that datacenter across all services. Answer with the "
            "number only."
        ),
        # Fenna -> region 4 -> dc-delta -> incident 104 -> 90 minutes
        "checker": equals_any("90", "90 minutes"),
        "expected": "90",
    },
    {
        "id": "A4_cross_filter",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 4,
        "prompt": (
            "Using the tools, consider every employee in the 'billing' "
            "department. Collect the datacenters of the regions they work in. "
            "Report the total number of severity-1 incidents that occurred at "
            "those datacenters. Answer with the number only."
        ),
        # billing: Cai(r1->dc-alpha), Dara(r2->dc-beta), Gus(r2->dc-beta),
        # Jo(r2->dc-beta) => {dc-alpha, dc-beta}; sev1: 102 (beta), 106 (alpha) = 2
        "checker": equals_any("2"),
        "expected": "2",
    },
    # --- repo: locate a real off-by-one in a tiny source tree -------------
    {
        "id": "B1_find_bug_file",
        "family": "repo",
        "tools": REPO_TOOLS,
        "min_calls": 2,
        "prompt": (
            "This repository has a failing test in tests/test_pricing.py. "
            "Using the tools, read the code, work out why the test fails, and "
            "report the path of the file containing the defect and the name of "
            "the function containing it."
        ),
        "checker": contains_all("src/pricing.py", "tier_rate"),
        "expected": "src/pricing.py, tier_rate",
    },
    {
        "id": "B2_state_the_fix",
        "family": "repo",
        "tools": REPO_TOOLS,
        "min_calls": 2,
        "prompt": (
            "This repository has a failing test in tests/test_pricing.py. "
            "Using the tools, find the single line that must change to make it "
            "pass, and report the corrected line of code exactly as it should "
            "appear, with no explanation."
        ),
        "checker": lambda ans: (
            ans is not None
            and ">=" in ans
            and "subtotal_cents" in ans
            and "threshold" in ans
        ),
        "expected": REPO_FIX_LINE,
    },
]


# --- harder tier -----------------------------------------------------------
# The first six tasks are solved by every capable model, so they measure
# "does tool calling work at all". These separate the field: longer chains,
# an ambiguous lookup that punishes guessing, and a defect whose obvious
# candidate fix is wrong.

TASKS_HARD = [
    {
        "id": "H1_chain_all_managers",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 5,
        "prompt": (
            "Using the tools, find every employee in the 'search' department. "
            "For each of them, follow the manager chain all the way up until "
            "an employee with no manager is reached. Report the name of that "
            "top employee, and the total number of DISTINCT employees you had "
            "to look at across all those chains, counting the search employees "
            "themselves and the top employee. Answer in the form: "
            "<name> / <number>"
        ),
        # search: Emil(5, mgr 8), Fenna(6, mgr 8) -> Hana(8, mgr 1) -> Ada(1, mgr None)
        # distinct: 5,6,8,1 = 4
        "checker": lambda ans: (
            ans is not None
            and _norm("ada speirs") in _norm(ans)
            and re.search(r"\b4\b", ans) is not None
        ),
        "expected": "Ada Speirs / 4",
    },
    {
        "id": "H2_ambiguous_lookup",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 2,
        "prompt": (
            "Using the tools, look up the employee whose name contains 'Jo' "
            "and report the datacenter of their region. If more than one "
            "employee matches, report the datacenter for EACH matching "
            "employee, as '<employee name>=<datacenter>' pairs separated by "
            "semicolons, sorted by employee id."
        ),
        # The trap: 'Jo' matches Hana Okonjo (id 8, r4 -> dc-delta) as well as
        # the obvious Jo Lindqvist (id 10, r2 -> dc-beta). A model that assumes
        # one match, or that never checks, gets this wrong.
        "checker": contains_all("Hana Okonjo", "dc-delta", "Jo Lindqvist", "dc-beta"),
        "expected": "Hana Okonjo=dc-delta; Jo Lindqvist=dc-beta",
    },
    {
        "id": "H3_worst_region",
        "family": "db",
        "tools": DB_TOOLS,
        "min_calls": 4,
        "prompt": (
            "Using the tools, work out which REGION NAME has the highest total "
            "incident minutes at its datacenter, counting only incidents of "
            "severity 2 or worse (severity number 2 or lower). Report the "
            "region name only."
        ),
        # sev<=2 by dc: alpha 45+240=285, beta 180+60=240, gamma 0, delta 90
        # alpha -> region 1 -> Nordwest
        "checker": equals_any("Nordwest", "the Nordwest region", "Nordwest region"),
        "expected": "Nordwest",
    },
    {
        "id": "H4_distractor_fix",
        "family": "repo",
        "tools": REPO_TOOLS,
        "min_calls": 3,
        "prompt": (
            "In this repository, tests/test_pricing.py fails. A colleague "
            "proposes fixing it by changing apply_discount in src/pricing.py "
            "to use round() instead of int() when converting to whole cents. "
            "Using the tools, decide whether that proposal makes the failing "
            "test pass. Answer YES or NO, then after a comma name the function "
            "that should actually be changed."
        ),
        # The proposal touches rounding in apply_discount, which the failing
        # test never calls: the test only exercises tier_rate. A model that
        # pattern-matches "pricing bug -> rounding" without reading the test
        # answers YES.
        "checker": lambda ans: (
            ans is not None
            and _norm(ans).startswith("no")
            and "tier_rate" in (ans or "")
        ),
        "expected": "NO, tier_rate",
    },
]

TASKS = TASKS + TASKS_HARD

for t in TASKS:
    t["prompt"] = t["prompt"] + ANSWER_INSTR

"""A small, fully deterministic tool environment for the agentic eval.

Nothing here executes model-written code. Tools read from fixed in-memory data,
so a task either was solved by correct multi-step tool use or it was not, and
the grader is a pure function over the final answer.

Two task families:
  * `db`   - multi-hop lookups across three tables; needs 3-5 chained calls
  * `repo` - navigate a small source tree, find the defect, name the fix
"""

import json

# --------------------------------------------------------------------------
# db world
# --------------------------------------------------------------------------

EMPLOYEES = [
    {"id": 1, "name": "Ada Speirs", "dept": "platform", "manager_id": None, "region_id": 3},
    {"id": 2, "name": "Bo Ferrante", "dept": "platform", "manager_id": 1, "region_id": 3},
    {"id": 3, "name": "Cai Mwangi", "dept": "billing", "manager_id": 7, "region_id": 1},
    {"id": 4, "name": "Dara Olsen", "dept": "billing", "manager_id": 7, "region_id": 2},
    {"id": 5, "name": "Emil Vance", "dept": "search", "manager_id": 8, "region_id": 1},
    {"id": 6, "name": "Fenna Brill", "dept": "search", "manager_id": 8, "region_id": 4},
    {"id": 7, "name": "Gus Adeyemi", "dept": "billing", "manager_id": 1, "region_id": 2},
    {"id": 8, "name": "Hana Okonjo", "dept": "search", "manager_id": 1, "region_id": 4},
    {"id": 9, "name": "Ivo Petran", "dept": "platform", "manager_id": 2, "region_id": 3},
    {"id": 10, "name": "Jo Lindqvist", "dept": "billing", "manager_id": 4, "region_id": 2},
]

REGIONS = {
    1: {"id": 1, "name": "Nordwest", "datacenter": "dc-alpha", "timezone": "UTC+1"},
    2: {"id": 2, "name": "Sudost", "datacenter": "dc-beta", "timezone": "UTC+2"},
    3: {"id": 3, "name": "Kustland", "datacenter": "dc-gamma", "timezone": "UTC+0"},
    4: {"id": 4, "name": "Hochtal", "datacenter": "dc-delta", "timezone": "UTC+3"},
}

INCIDENTS = [
    {"id": 101, "datacenter": "dc-alpha", "severity": 2, "minutes": 45, "service": "checkout"},
    {"id": 102, "datacenter": "dc-beta", "severity": 1, "minutes": 180, "service": "checkout"},
    {"id": 103, "datacenter": "dc-gamma", "severity": 3, "minutes": 15, "service": "index"},
    {"id": 104, "datacenter": "dc-delta", "severity": 2, "minutes": 90, "service": "index"},
    {"id": 105, "datacenter": "dc-beta", "severity": 2, "minutes": 60, "service": "ledger"},
    {"id": 106, "datacenter": "dc-alpha", "severity": 1, "minutes": 240, "service": "ledger"},
]

# --------------------------------------------------------------------------
# repo world
# --------------------------------------------------------------------------

REPO = {
    "README.md": "billing-core\n============\nRounds and applies tiered discounts.\n",
    "src/__init__.py": "",
    "src/pricing.py": '''"""Tiered discount pricing."""

TIERS = [
    (0, 0.00),
    (1000, 0.05),
    (5000, 0.10),
    (20000, 0.15),
]


def tier_rate(subtotal_cents):
    """Return the discount rate for a subtotal."""
    rate = 0.0
    for threshold, r in TIERS:
        if subtotal_cents > threshold:
            rate = r
    return rate


def apply_discount(subtotal_cents):
    """Apply the tier discount and round to whole cents."""
    rate = tier_rate(subtotal_cents)
    return int(subtotal_cents * (1 - rate))
''',
    "src/invoice.py": '''from src.pricing import apply_discount


def total_for(lines):
    subtotal = sum(l["qty"] * l["unit_cents"] for l in lines)
    return apply_discount(subtotal)
''',
    "tests/test_pricing.py": '''from src.pricing import tier_rate


def test_boundaries():
    assert tier_rate(999) == 0.00
    assert tier_rate(1000) == 0.05
    assert tier_rate(4999) == 0.05
    assert tier_rate(5000) == 0.10
''',
}

# tier_rate uses `>` so a subtotal exactly equal to a threshold falls into the
# band below it: tier_rate(1000) returns 0.00, and the test expects 0.05.
REPO_BUG_FILE = "src/pricing.py"
REPO_BUG_LINE = "if subtotal_cents > threshold:"
REPO_FIX_LINE = "if subtotal_cents >= threshold:"


# --------------------------------------------------------------------------
# tool schemas
# --------------------------------------------------------------------------

DB_TOOLS = [
    {"type": "function", "function": {
        "name": "find_employee",
        "description": "Look up employees by name substring or by department.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "case-insensitive substring of the employee name"},
            "dept": {"type": "string", "description": "exact department name"},
        }, "required": []}}},
    {"type": "function", "function": {
        "name": "get_employee",
        "description": "Get one employee record by numeric id.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer"}}, "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "get_region",
        "description": "Get a region record (name, datacenter, timezone) by region id.",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer"}}, "required": ["id"]}}},
    {"type": "function", "function": {
        "name": "list_incidents",
        "description": "List incidents, optionally filtered by datacenter and/or service.",
        "parameters": {"type": "object", "properties": {
            "datacenter": {"type": "string"},
            "service": {"type": "string"}}, "required": []}}},
]

REPO_TOOLS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List every file path in the repository.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the full text of one file in the repository.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search the repository for a substring; returns path and line for each hit.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}}, "required": ["pattern"]}}},
]


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------

def call_tool(name, args):
    if not isinstance(args, dict):
        return {"error": f"arguments must be an object, got {type(args).__name__}"}

    if name == "find_employee":
        rows = EMPLOYEES
        if args.get("name"):
            q = str(args["name"]).lower()
            rows = [e for e in rows if q in e["name"].lower()]
        if args.get("dept"):
            rows = [e for e in rows if e["dept"] == args["dept"]]
        return {"employees": rows}

    if name == "get_employee":
        try:
            i = int(args["id"])
        except (KeyError, TypeError, ValueError):
            return {"error": "id must be an integer"}
        for e in EMPLOYEES:
            if e["id"] == i:
                return {"employee": e}
        return {"error": f"no employee with id {i}"}

    if name == "get_region":
        try:
            i = int(args["id"])
        except (KeyError, TypeError, ValueError):
            return {"error": "id must be an integer"}
        r = REGIONS.get(i)
        return {"region": r} if r else {"error": f"no region with id {i}"}

    if name == "list_incidents":
        rows = INCIDENTS
        if args.get("datacenter"):
            rows = [x for x in rows if x["datacenter"] == args["datacenter"]]
        if args.get("service"):
            rows = [x for x in rows if x["service"] == args["service"]]
        return {"incidents": rows}

    if name == "list_files":
        return {"files": sorted(REPO)}

    if name == "read_file":
        p = str(args.get("path", "")).replace("\\", "/").lstrip("./")
        if p in REPO:
            return {"path": p, "content": REPO[p]}
        return {"error": f"no such file: {args.get('path')}",
                "available": sorted(REPO)}

    if name == "grep":
        pat = str(args.get("pattern", ""))
        hits = []
        for path, text in sorted(REPO.items()):
            for n, line in enumerate(text.splitlines(), 1):
                if pat in line:
                    hits.append({"path": path, "line": n, "text": line})
        return {"hits": hits}

    return {"error": f"unknown tool: {name}"}


def tool_result_text(name, args):
    return json.dumps(call_tool(name, args))

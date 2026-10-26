"""Harder agentic environment: long horizon, pagination, recovery, and traps.

The Ornith round's agentic eval returned 18/18 for all five models, so it
measured nothing.  Three things are added here, each aimed at a distinct
failure mode rather than at raw difficulty:

  pagination  - `list_shipments` returns at most 3 rows and a cursor, so a
                complete answer needs repeated calls.  A model that answers
                from the first page gets a plausible wrong number.
  recovery    - `open_manifest` rejects an unpadded id once, with a message
                saying exactly how to fix it.  The task is only solvable by
                reading the error and retrying.
  traps       - two tasks ask for data the world does not contain.  The only
                correct reply is ANSWER: UNAVAILABLE.  Graded alongside the
                solvable tasks so a model cannot win by refusing everything.

Everything is deterministic and in memory.  No model-written code is executed.
"""

import json
import re

# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------

WAREHOUSES = {
    "WH-01": {"city": "Torvik", "manager": "R. Adeyemi", "dock_count": 4},
    "WH-02": {"city": "Halden", "manager": "M. Ferreira", "dock_count": 7},
    "WH-03": {"city": "Ostrava", "manager": "K. Nowak", "dock_count": 2},
}

# 11 shipments so a 3-per-page cursor needs four calls for the biggest set
SHIPMENTS = [
    {"id": "SH-1001", "warehouse": "WH-01", "carrier": "Delkin", "weight_kg": 420, "status": "delivered"},
    {"id": "SH-1002", "warehouse": "WH-01", "carrier": "Brightway", "weight_kg": 130, "status": "in_transit"},
    {"id": "SH-1003", "warehouse": "WH-02", "carrier": "Delkin", "weight_kg": 980, "status": "delivered"},
    {"id": "SH-1004", "warehouse": "WH-01", "carrier": "Delkin", "weight_kg": 275, "status": "delayed"},
    {"id": "SH-1005", "warehouse": "WH-03", "carrier": "Corvid", "weight_kg": 55, "status": "delivered"},
    {"id": "SH-1006", "warehouse": "WH-01", "carrier": "Corvid", "weight_kg": 610, "status": "delivered"},
    {"id": "SH-1007", "warehouse": "WH-02", "carrier": "Brightway", "weight_kg": 340, "status": "in_transit"},
    {"id": "SH-1008", "warehouse": "WH-01", "carrier": "Delkin", "weight_kg": 190, "status": "delivered"},
    {"id": "SH-1009", "warehouse": "WH-03", "carrier": "Delkin", "weight_kg": 720, "status": "delayed"},
    {"id": "SH-1010", "warehouse": "WH-01", "carrier": "Brightway", "weight_kg": 45, "status": "delivered"},
    {"id": "SH-1011", "warehouse": "WH-02", "carrier": "Corvid", "weight_kg": 505, "status": "delivered"},
]

# manifests are keyed by a zero-padded 6-digit form of the shipment number
MANIFESTS = {
    "001001": {"pallets": 6, "hazmat": False, "seal": "S-4471"},
    "001003": {"pallets": 14, "hazmat": True, "seal": "S-8820"},
    "001006": {"pallets": 9, "hazmat": False, "seal": "S-1290"},
    "001009": {"pallets": 11, "hazmat": True, "seal": "S-6634"},
    "001011": {"pallets": 7, "hazmat": False, "seal": "S-3005"},
}

CARRIER_RATES = {
    "Delkin": {"per_kg": 0.42, "minimum": 60.0},
    "Brightway": {"per_kg": 0.55, "minimum": 40.0},
    "Corvid": {"per_kg": 0.31, "minimum": 95.0},
}

PAGE = 3


class World:
    """Per-episode state, so the retry trap resets between tasks."""

    def __init__(self):
        self.rejected_once = set()
        self.calls = []

    # -- tools ------------------------------------------------------------
    def list_warehouses(self):
        return {"warehouses": sorted(WAREHOUSES)}

    def get_warehouse(self, warehouse_id):
        w = WAREHOUSES.get(warehouse_id)
        if not w:
            return {"error": f"unknown warehouse {warehouse_id}"}
        return dict(w, id=warehouse_id)

    def list_shipments(self, warehouse_id=None, cursor=0):
        rows = [s for s in SHIPMENTS
                if warehouse_id is None or s["warehouse"] == warehouse_id]
        try:
            cursor = int(cursor)
        except (TypeError, ValueError):
            cursor = 0
        page = rows[cursor:cursor + PAGE]
        nxt = cursor + PAGE if cursor + PAGE < len(rows) else None
        return {"shipments": page, "next_cursor": nxt,
                "note": "at most 3 rows per call; call again with next_cursor "
                        "until next_cursor is null"}

    def open_manifest(self, shipment_id):
        """Requires the 6-digit padded numeric id, and says so once."""
        sid = str(shipment_id)
        if not re.fullmatch(r"\d{6}", sid):
            if sid not in self.rejected_once:
                self.rejected_once.add(sid)
                return {"error": "invalid manifest key",
                        "detail": "manifest keys are the shipment number "
                                  "zero-padded to 6 digits, so shipment "
                                  "SH-1002 is manifest key 001002. Retry with "
                                  "that form."}
            return {"error": "invalid manifest key"}
        m = MANIFESTS.get(sid)
        if not m:
            return {"error": f"no manifest on file for key {sid}"}
        return dict(m, key=sid)

    def carrier_rate(self, carrier):
        r = CARRIER_RATES.get(carrier)
        if not r:
            return {"error": f"unknown carrier {carrier}"}
        return dict(r, carrier=carrier)


TOOLS = [
    {"type": "function", "function": {
        "name": "list_warehouses",
        "description": "List all warehouse ids.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_warehouse",
        "description": "Details for one warehouse: city, manager, dock count.",
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string"}}, "required": ["warehouse_id"]}}},
    {"type": "function", "function": {
        "name": "list_shipments",
        "description": ("List shipments, optionally filtered by warehouse. "
                        "Returns at most 3 per call plus next_cursor; call "
                        "again with the cursor until next_cursor is null."),
        "parameters": {"type": "object", "properties": {
            "warehouse_id": {"type": "string"},
            "cursor": {"type": "integer"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "open_manifest",
        "description": "Open the manifest for a shipment by its manifest key.",
        "parameters": {"type": "object", "properties": {
            "shipment_id": {"type": "string"}}, "required": ["shipment_id"]}}},
    {"type": "function", "function": {
        "name": "carrier_rate",
        "description": "Per-kg rate and minimum charge for a carrier.",
        "parameters": {"type": "object", "properties": {
            "carrier": {"type": "string"}}, "required": ["carrier"]}}},
]


def dispatch(world, name, args):
    fn = getattr(world, name, None)
    if fn is None:
        return {"error": f"no such tool {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}


# --------------------------------------------------------------------------
# graders
# --------------------------------------------------------------------------

def _norm(s):
    return re.sub(r"[^a-z0-9.]+", " ", (s or "").lower()).strip()


def num_within(expected, tol=0.011):
    def check(ans):
        m = re.findall(r"-?\d+(?:\.\d+)?", (ans or "").replace(",", ""))
        if not m:
            return False
        try:
            return abs(float(m[-1]) - expected) <= tol
        except ValueError:
            return False
    return check


def equals_any(*accepted):
    acc = {_norm(a) for a in accepted}

    def check(ans):
        return _norm(ans) in acc
    return check


def is_unavailable(ans):
    a = _norm(ans).replace(" ", "")
    return any(w in a for w in ("unavailable", "notavailable", "noinformation",
                                "notinthedata", "unknown", "cannotbedetermined",
                                "nodata"))


# --------------------------------------------------------------------------
# tasks.  Every expected value is computed from the world above, not typed in.
# --------------------------------------------------------------------------

def _wh01_delivered_weight():
    return sum(s["weight_kg"] for s in SHIPMENTS
               if s["warehouse"] == "WH-01" and s["status"] == "delivered")


def _delkin_cost_wh01():
    total = 0.0
    for s in SHIPMENTS:
        if s["warehouse"] == "WH-01" and s["carrier"] == "Delkin":
            rate = CARRIER_RATES["Delkin"]
            total += max(rate["minimum"], s["weight_kg"] * rate["per_kg"])
    return round(total, 2)


def _hazmat_pallets():
    return sum(m["pallets"] for m in MANIFESTS.values() if m["hazmat"])


def _heaviest_delayed_manager():
    delayed = [s for s in SHIPMENTS if s["status"] == "delayed"]
    top = max(delayed, key=lambda s: s["weight_kg"])
    return WAREHOUSES[top["warehouse"]]["manager"]


TASKS = [
    dict(id="H1", kind="pagination", hops=5,
         prompt="Using the tools, find the total weight in kilograms of all "
                "shipments at warehouse WH-01 whose status is delivered. "
                "Report the number only.",
         checker=num_within(_wh01_delivered_weight())),
    dict(id="H2", kind="pagination+rate", hops=6,
         prompt="Using the tools, compute the total shipping cost for every "
                "Delkin shipment at warehouse WH-01. Each shipment costs its "
                "weight times the carrier per-kg rate, or the carrier minimum "
                "charge if that is larger. Report the total in currency units, "
                "to two decimal places.",
         checker=num_within(_delkin_cost_wh01(), tol=0.02)),
    dict(id="H3", kind="recovery", hops=3,
         prompt="Using the tools, open the manifest for shipment SH-1003 and "
                "report its seal number.",
         checker=equals_any("S-8820", "S8820")),
    dict(id="H4", kind="recovery+aggregate", hops=7,
         prompt="Using the tools, check the manifest of every shipment in "
                "every warehouse. Some shipments have no manifest on file; "
                "ignore those. Total the pallets across the manifests that "
                "are marked hazmat. Report the number only.",
         checker=num_within(_hazmat_pallets())),
    dict(id="H5", kind="multi-hop", hops=4,
         prompt="Using the tools, identify the manager of the warehouse that "
                "holds the heaviest shipment with status delayed. Report the "
                "manager's name exactly as it appears in the data.",
         checker=equals_any(_heaviest_delayed_manager(), "K. Nowak", "K Nowak")),
    dict(id="H6", kind="trap", hops=2,
         prompt="Using the tools, report the insurance policy number covering "
                "shipment SH-1006.",
         checker=is_unavailable),
    dict(id="H7", kind="trap", hops=3,
         prompt="Using the tools, report the manifest seal number for shipment "
                "SH-1002.",
         checker=is_unavailable),
    dict(id="H8", kind="multi-hop", hops=3,
         prompt="Using the tools, report how many loading docks the warehouse "
                "in the city of Ostrava has. Report the number only.",
         checker=num_within(2)),
]

ANSWER_INSTR = (
    "\n\nWork by calling the tools. When you are certain, reply with a final "
    "line in exactly this format and nothing after it:\nANSWER: <your answer>\n"
    "If the tools do not contain the information asked for, reply "
    "ANSWER: UNAVAILABLE rather than guessing."
)


def extract(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    return m[-1].strip() if m else None


if __name__ == "__main__":
    print("expected values derived from the world:")
    print("  H1 delivered weight WH-01 :", _wh01_delivered_weight())
    print("  H2 Delkin cost WH-01      :", _delkin_cost_wh01())
    print("  H4 hazmat pallets         :", _hazmat_pallets())
    print("  H5 heaviest delayed mgr   :", _heaviest_delayed_manager())
    w = World()
    print("  pagination pages for WH-01:",
          json.dumps(w.list_shipments("WH-01"))[:120])
    print("  retry trap first reply    :",
          json.dumps(w.open_manifest("SH-1003"))[:120])
    print("  retry trap second reply   :",
          json.dumps(w.open_manifest("001003"))[:120])

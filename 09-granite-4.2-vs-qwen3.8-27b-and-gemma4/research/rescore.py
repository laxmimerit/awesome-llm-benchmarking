"""Re-apply the graders to stored answers, without re-running any model.

Two trap items were wrong when the sweep ran, and the models were right:

  T8 - "a train leaves the station ... when does the second catch the first"
       was written as a trap on the grounds that direction and separation were
       never stated.  Leaving *the* station implies a shared origin and
       direction, which makes the chase well posed, and all six models
       answered 7:00 pm, which is correct.  Reclassified as a solvable item.
  T6 - drawing 8 balls from a bag of 7.  Answering that the probability is 0
       is defensible for an impossible event rather than a hallucination, so 0
       is now accepted alongside IMPOSSIBLE.

Correctness is a pure function of the extracted answer, which is stored in
every row, so the fix is a rescore rather than another five GPU hours.  This
script writes a new file and leaves the original untouched.
"""

import json
from pathlib import Path

from problems_hard import ALL_ITEMS, extract_answer

RESULTS = Path(__file__).resolve().parent.parent / "results"
BY_ID = {i["id"]: i for i in ALL_ITEMS}


def rescore(row):
    item = BY_ID.get(row["id"])
    if item is None:
        return row
    out = dict(row)
    out["dept_original"] = row["dept"]
    out["correct_original"] = row["correct"]
    out["dept"] = item["dept"]

    if item["dept"] == "format":
        return out                      # graded on full text, not re-derivable
    ans = row.get("answer_raw") or None
    out["correct"] = bool(ans and item["checker"](ans))
    out["changed"] = out["correct"] != row["correct"] or out["dept"] != row["dept"]
    return out


def main():
    src = RESULTS / "quality_hard.jsonl"
    dst = RESULTS / "quality_hard_rescored.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = [rescore(r) for r in rows]
    changed = [r for r in out if r.get("changed")]
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"{len(rows)} rows, {len(changed)} changed by the grader fix")
    for r in changed:
        print(f"  {r['model']:16} {r['id']:4} "
              f"{r['dept_original']}->{r['dept']} "
              f"{r['correct_original']}->{r['correct']} "
              f"answer={r.get('answer_raw','')[:30]!r}")
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()

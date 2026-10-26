"""Analysis over results.jsonl: accuracy, thinking economics, direction of thinking, speed."""
import json, re, statistics as st
from collections import defaultdict

BASE = r"D:\Claude AI\AI Model Performance\qwen-teardown"
MODELS = ["qwen3.5:27b", "qwen3.6:27b", "qwen3.8:latest"]
SHORT = {"qwen3.5:27b": "3.5", "qwen3.6:27b": "3.6", "qwen3.8:latest": "3.8"}


def load(path=BASE + r"\results.jsonl"):
    best = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        k = (r["key"], r.get("draft"))
        if k not in best or (best[k].get("error") and not r.get("error")):
            best[k] = r
    return list(best.values())


def tps(r):
    if r.get("eval_count") and r.get("eval_duration_ns"):
        return r["eval_count"] / (r["eval_duration_ns"] / 1e9)


def prefill_tps(r):
    if r.get("prompt_eval_count") and r.get("prompt_eval_duration_ns"):
        return r["prompt_eval_count"] / (r["prompt_eval_duration_ns"] / 1e9)


SELFCORR = re.compile(r"\b(wait|hmm|actually|let me re-?(check|consider|examine|verify)|i made (a|an) (mistake|error)|that's (wrong|not right)|hold on)\b", re.I)
VERIFY = re.compile(r"\b(verify|double-?check|confirm|sanity check|check(ing)? (this|that|each|the))\b", re.I)


def think_stats(r):
    t = r.get("thinking") or ""
    toks = r.get("eval_count") or 0
    n_sc = len(SELFCORR.findall(t))
    n_vf = len(VERIFY.findall(t))
    return dict(chars=len(t), self_corr=n_sc, verify=n_vf)


def table(rows, title):
    print("\n== " + title + " ==")
    for row in rows:
        print("  " + " | ".join(str(c) for c in row))


def main():
    rows = load()
    by = defaultdict(list)
    for r in rows:
        by[r["pass_name"].split("_")[0]].append(r)

    # ---- Pass A: matched accuracy + thinking economics ----
    for mode, label in [(True, "think"), (False, "nothink")]:
        out = [["model", "acc", "avg_think_chars", "avg_eval_tok", "avg_wall_s", "selfcorr/run", "verify/run"]]
        for m in MODELS:
            rs = [r for r in by.get("A", []) if r["model"] == m and r["think"] == mode and not r.get("error")]
            if not rs:
                continue
            accs = sum(bool(r.get("correct")) for r in rs)
            ts = [think_stats(r) for r in rs]
            out.append([SHORT[m], f"{accs}/{len(rs)}",
                        round(st.mean(t["chars"] for t in ts)),
                        round(st.mean(r["eval_count"] for r in rs if r.get("eval_count"))),
                        round(st.mean(r["wall_s"] for r in rs), 1),
                        round(st.mean(t["self_corr"] for t in ts), 1),
                        round(st.mean(t["verify"] for t in ts), 1)])
        table(out, f"Pass A matched ({label})")

    # per-problem thinking tokens (think mode)
    out = [["problem"] + [SHORT[m] for m in MODELS]]
    probs = sorted({r["problem_id"] for r in by.get("A", []) if r["problem_id"] != "SPEED"})
    for p in probs:
        row = [p]
        for m in MODELS:
            rs = [r for r in by["A"] if r["model"] == m and r["think"] is True and r["problem_id"] == p and r.get("eval_count")]
            marks = "".join("x" if not r.get("correct") else "" for r in rs)
            row.append((str(round(st.mean(r["eval_count"] for r in rs))) if rs else "-") + marks)
        out.append(row)
    table(out, "Pass A thinking tokens per problem (x = a wrong run)")

    # ---- Pass C vs A (recommended vs matched) ----
    out = [["model", "A_acc", "C_acc", "A_think_tok", "C_think_tok"]]
    for m in MODELS:
        ra = [r for r in by.get("A", []) if r["model"] == m and r["think"] is True and not r.get("error")]
        rc = [r for r in by.get("C", []) if r["model"] == m and not r.get("error")]
        if not rc:
            continue
        out.append([SHORT[m],
                    f"{sum(bool(r.get('correct')) for r in ra)}/{len(ra)}",
                    f"{sum(bool(r.get('correct')) for r in rc)}/{len(rc)}",
                    round(st.mean(r["eval_count"] for r in ra if r.get("eval_count"))),
                    round(st.mean(r["eval_count"] for r in rc if r.get("eval_count")))])
    table(out, "Pass C recommended settings vs Pass A matched (thinking)")

    # ---- Pass E: effort ladder ----
    out = [["effort", "acc", "avg_think_chars", "avg_eval_tok", "avg_wall_s"]]
    for lvl in ["low", "medium", "high", "max"]:
        rs = [r for r in by.get("E", []) if r["think"] == lvl and not r.get("error")]
        if not rs:
            continue
        out.append([lvl, f"{sum(bool(r.get('correct')) for r in rs)}/{len(rs)}",
                    round(st.mean(len(r.get('thinking') or '') for r in rs)),
                    round(st.mean(r["eval_count"] for r in rs if r.get("eval_count"))),
                    round(st.mean(r["wall_s"] for r in rs), 1)])
    table(out, "Pass E qwen3.8 effort ladder (8 problems each)")

    # ---- Pass F: speed probes ----
    out = [["model", "mtp", "order", "tok/s (mean of 3)", "temp_before"]]
    for m in MODELS:
        for draft in [0, 4]:
            for order in ["fwd", "rev"]:
                rs = [r for r in by.get("F", []) if r["model"] == m and r["draft"] == draft and r["order"] == order and tps(r)]
                if not rs:
                    continue
                out.append([SHORT[m], "on" if draft else "off", order,
                            round(st.mean(tps(r) for r in rs), 1),
                            round(st.mean(r["gpu_before"].get("temp", 0) for r in rs))])
    table(out, "Pass F decode speed (600-token essays)")

    # ---- Pass D: context ladder ----
    if by.get("D"):
        out = [["model", "fill_target", "problem", "acc(n)", "prompt_tok", "prefill_tok/s", "decode_tok/s", "think_chars"]]
        for m in MODELS:
            for fill in sorted({r["fill"] for r in by["D"]}):
                for p in ["M1_powers", "R3_houses"]:
                    rs = [r for r in by["D"] if r["model"] == m and r["fill"] == fill and r["problem_id"] == p and not r.get("error")]
                    if not rs:
                        continue
                    out.append([SHORT[m], fill, p[:2],
                                f"{sum(bool(r.get('correct')) for r in rs)}/{len(rs)}",
                                rs[0].get("prompt_eval_count"),
                                round(st.mean(prefill_tps(r) for r in rs if prefill_tps(r))),
                                round(st.mean(tps(r) for r in rs if tps(r)), 1),
                                round(st.mean(len(r.get("thinking") or "") for r in rs))])
        table(out, "Pass D context ladder")

    # ---- Pass G: hard tier ----
    if by.get("G"):
        out = [["model", "think", "problem", "correct(n)", "think_chars", "eval_tok", "wall_s"]]
        for m in MODELS + ["qwen3.8:latest"]:
            seen = set()
            for r in sorted(by["G"], key=lambda r: r["key"]):
                if r["model"] != m or r.get("error"):
                    continue
                k = (r["model"], str(r["think"]), r["problem_id"])
                if k in seen:
                    continue
                seen.add(k)
                rs = [x for x in by["G"] if (x["model"], str(x["think"]), x["problem_id"]) == k and not x.get("error")]
                out.append([SHORT[m], str(r["think"]), r["problem_id"][:2],
                            f"{sum(bool(x.get('correct')) for x in rs)}/{len(rs)}",
                            round(st.mean(len(x.get("thinking") or "") for x in rs)),
                            round(st.mean(x["eval_count"] for x in rs if x.get("eval_count"))),
                            round(st.mean(x["wall_s"] for x in rs), 1)])
            if m == "qwen3.6:27b":
                pass
        table(out, "Pass G hard tier")

    # order effect check (pass A think): fwd vs rev wall time
    out = [["model", "fwd tok/s", "rev tok/s", "delta%"]]
    for m in MODELS:
        f = [tps(r) for r in by.get("A", []) if r["model"] == m and r["order"] == "fwd" and tps(r)]
        v = [tps(r) for r in by.get("A", []) if r["model"] == m and r["order"] == "rev" and tps(r)]
        if f and v:
            mf, mv = st.mean(f), st.mean(v)
            out.append([SHORT[m], round(mf, 1), round(mv, 1), round(100 * (mf - mv) / mv, 1)])
    table(out, "Order effect on decode speed (pass A, all runs)")


if __name__ == "__main__":
    main()

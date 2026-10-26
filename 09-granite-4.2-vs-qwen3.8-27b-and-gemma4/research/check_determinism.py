"""Control for the MTP output-identity test.

If two greedy runs of the SAME configuration already diverge, then "MTP changed
the output" is not a finding about MTP. This measures the baseline first, then
reports where the MTP run actually parts company.
"""

import argparse
import json
from pathlib import Path

import server
from workloads import DECODE_PROMPTS

RESULTS = Path(__file__).resolve().parent.parent / "results"


def greedy(prompt, n_predict, seed=7):
    return server.completion(
        prompt, n_predict=n_predict,
        sampling={"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                  "min_p": 0.0, "presence_penalty": 0.0},
        seed=seed, ignore_eos=True, cache_prompt=False,
    )["content"]


def divergence_point(a, b):
    """First differing character index, and a short window around it."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, a[max(0, i - 60):i + 40], b[max(0, i - 60):i + 40]
    if len(a) != len(b):
        return n, a[max(0, n - 60):], b[max(0, n - 60):]
    return None, "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--contents", nargs="*", default=["counting", "code", "essay"])
    ap.add_argument("--out", default="determinism.json")
    args = ap.parse_args()

    out = {"model": args.model, "n_predict": args.n_predict, "cases": {}}

    # --- baseline: same server, same config, two runs -----------------
    print("=== control: MTP off, two identical greedy runs ===", flush=True)
    srv = server.Server(args.model, ctx=8192, tag=f"{args.model}_det_off")
    srv.start()
    off_texts = {}
    try:
        greedy(DECODE_PROMPTS["counting"], 32)
        for c in args.contents:
            t1 = greedy(DECODE_PROMPTS[c], args.n_predict)
            t2 = greedy(DECODE_PROMPTS[c], args.n_predict)
            off_texts[c] = t1
            idx, wa, wb = divergence_point(t1, t2)
            out["cases"].setdefault(c, {})["self_identical"] = (idx is None)
            out["cases"][c]["self_divergence_char"] = idx
            print(f"  {c:9} run1 vs run2 identical={idx is None} "
                  f"(first diff at char {idx})", flush=True)
    finally:
        srv.stop()

    # --- MTP on, compared against the baseline text -------------------
    print("\n=== MTP on, compared to the MTP-off text ===", flush=True)
    srv = server.Server(args.model, ctx=8192, tag=f"{args.model}_det_on",
                        spec_type="draft-mtp", spec_n_max=4)
    srv.start()
    try:
        greedy(DECODE_PROMPTS["counting"], 32)
        for c in args.contents:
            t = greedy(DECODE_PROMPTS[c], args.n_predict)
            idx, wa, wb = divergence_point(off_texts[c], t)
            out["cases"][c]["mtp_identical"] = (idx is None)
            out["cases"][c]["mtp_divergence_char"] = idx
            out["cases"][c]["mtp_divergence_frac"] = (
                round(idx / max(1, len(off_texts[c])), 3) if idx is not None else None
            )
            out["cases"][c]["window_off"] = wa
            out["cases"][c]["window_mtp"] = wb
            print(f"  {c:9} identical={idx is None} first diff at char {idx} "
                  f"of {len(off_texts[c])}", flush=True)
            if idx is not None:
                print(f"      off: ...{wa[-70:]!r}")
                print(f"      mtp: ...{wb[-70:]!r}")
    finally:
        srv.stop()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {RESULTS / args.out}")


if __name__ == "__main__":
    main()

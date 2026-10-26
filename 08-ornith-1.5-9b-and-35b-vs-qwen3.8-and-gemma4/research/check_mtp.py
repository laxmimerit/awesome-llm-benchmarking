"""Does the undocumented `nextn` head in the Ornith GGUFs actually work?

Neither Ornith model card mentions multi-token prediction, but both GGUFs carry
a blk.N.nextn.* block. llama.cpp ignores those tensors unless started with
--spec-type draft-mtp. This script checks three things in order:

  1. does the server accept the flag and stop calling the tensors "unused"
  2. does it produce identical text to the MTP-off run at temperature 0
  3. what is the draft acceptance rate, per content type

Step 2 is the one that matters. Speculative decoding is only a free speedup if
the output is bit-identical; if it is not, the speed number is not comparable.
"""

import argparse
import json
import re
import time
from pathlib import Path

import server
from models import REGISTRY
from workloads import DECODE_PROMPTS

RESULTS = Path(__file__).resolve().parent.parent / "results"

RE_UNUSED = re.compile(r"unused tensor (\S*nextn\S*)")
RE_DRAFT = re.compile(r"n_draft\s*=\s*(\d+)")
RE_ACCEPT = re.compile(r"accept\s*=\s*([\d.]+)")


def greedy(prompt, n_predict):
    """Temperature 0, fixed seed: the only setting where two runs must match."""
    return server.completion(
        prompt, n_predict=n_predict,
        sampling={"temperature": 0.0, "top_k": 1, "top_p": 1.0,
                  "min_p": 0.0, "presence_penalty": 0.0},
        seed=7, ignore_eos=True, cache_prompt=False,
    )


def run(model_key, spec_type, n_predict, contents):
    tag = f"{model_key}_mtp_{spec_type or 'off'}"
    srv = server.Server(model_key, ctx=8192, tag=tag,
                        spec_type=spec_type, spec_n_max=4 if spec_type else None,
                        extra=["-lv", "4"] if spec_type else None)
    srv.start()
    out = {"model": model_key, "spec_type": spec_type or "off",
           "vram_delta_mb": srv.loaded_mb - srv.baseline_mb, "runs": {}}
    try:
        # warm up so the first measured run is not building graphs
        greedy(DECODE_PROMPTS["counting"], 32)
        for content in contents:
            r = greedy(DECODE_PROMPTS[content], n_predict)
            t = r["timings"]
            out["runs"][content] = {
                "text": r["content"],
                "decode_tps": t.get("predicted_per_second"),
                "predicted_n": t["predicted_n"],
                "predicted_ms": t["predicted_ms"],
                "draft_n": t.get("draft_n"),
                "draft_n_accepted": t.get("draft_n_accepted"),
                "accept_rate": (
                    round(t["draft_n_accepted"] / t["draft_n"], 4)
                    if t.get("draft_n") else None
                ),
            }
            a = out["runs"][content]
            print(f"  {content:9} {a['decode_tps']:7.2f} tok/s"
                  + (f"  accept {a['accept_rate']}" if a["accept_rate"] is not None
                     else ""), flush=True)
    finally:
        srv.stop()
    log = Path(srv.log_path).read_text(errors="replace")
    out["unused_nextn_tensors"] = RE_UNUSED.findall(log)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--contents", nargs="*",
                    default=["counting", "code", "json", "essay", "fiction"])
    ap.add_argument("--out", default="mtp_check.json")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / args.out
    all_out = json.loads(out_path.read_text()) if out_path.exists() else {}

    for key in args.models:
        print(f"\n=== {key}: MTP off ===", flush=True)
        off = run(key, None, args.n_predict, args.contents)
        print(f"  unused nextn tensors reported: {len(off['unused_nextn_tensors'])}")

        print(f"=== {key}: MTP on (--spec-type draft-mtp) ===", flush=True)
        try:
            on = run(key, "draft-mtp", args.n_predict, args.contents)
        except Exception as e:
            print(f"  MTP FAILED TO START: {str(e)[:400]}", flush=True)
            all_out[key] = {"off": off, "on": None, "error": str(e)[:1000]}
            out_path.write_text(json.dumps(all_out, indent=2, default=str))
            continue

        identical = {}
        for content in args.contents:
            a = off["runs"][content]["text"]
            b = on["runs"][content]["text"]
            identical[content] = (a == b)
            speed = (on["runs"][content]["decode_tps"]
                     / off["runs"][content]["decode_tps"])
            print(f"  {content:9} identical={identical[content]!s:5} "
                  f"speedup={speed:5.2f}x "
                  f"accept={on['runs'][content]['accept_rate']}", flush=True)

        all_out[key] = {
            "off": off, "on": on, "identical": identical,
            "unused_nextn_off": off["unused_nextn_tensors"],
            "unused_nextn_on": on["unused_nextn_tensors"],
            "speedup": {
                c: round(on["runs"][c]["decode_tps"] / off["runs"][c]["decode_tps"], 4)
                for c in args.contents
            },
        }
        out_path.write_text(json.dumps(all_out, indent=2, default=str))
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

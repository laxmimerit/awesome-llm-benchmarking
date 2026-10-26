"""Compare the raw per-task records for Ollama vs llama.cpp at the same config.

A 2.7x throughput gap between two runtimes that share the same inference
engine is not credible, so this looks at what actually differs: token counts,
whether thinking was emitted, and how long generation really took.
"""

import json

d = json.load(open("results.json", encoding="utf-8"))

PAIR = [
    "ollama-ctx32k-kvf16-faon-specnone-think",
    "llamacpp-ctx32k-kvf16-faon-specnone-think",
]

for key in PAIR:
    v = d.get(key)
    if not v:
        print(f"{key}: not present\n")
        continue
    print(f"=== {key} ===")
    runs = [r for r in v.get("runs", []) if not r.get("error")]
    for t in ("decode_code", "decode_prose", "q_arithmetic"):
        for r in runs:
            if r["task"] != t or r.get("rep") != 0:
                continue
            print(
                f"  {t:<16} prompt_tok={r.get('prompt_tokens'):<6} "
                f"completion_tok={r.get('completion_tokens'):<6} "
                f"think_chunks={r.get('thinking_chunks'):<6} "
                f"stream_chunks={r.get('stream_chunks'):<6}"
            )
            print(
                f"  {'':<16} ttft={r.get('ttft_s'):<8} decode_s={r.get('decode_s'):<8} "
                f"wall={r.get('wall_s'):<8} decode_tps={r.get('decode_tps')}"
            )
            print(f"  {'':<16} head={r.get('response_head', '')[:90]!r}")
    print()

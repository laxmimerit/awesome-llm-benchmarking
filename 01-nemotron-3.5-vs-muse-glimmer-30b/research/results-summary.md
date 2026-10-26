# Local Benchmark Results (RTX 5090, Ollama, 64K context, 2026-08-11)

Both models run via Ollama, num_ctx=65536, temperature 0.2, single RTX 5090 (32GB).
- nemotron-3.5-lightning:latest (25 GB download, GGUF from Ollama registry — not the exact NVFP4 build)
- muse-glimmer:latest (18 GB download, 4-bit K-Quant)

## Accuracy — simple task suite (10 tasks)
Both models: **10/10** (9 auto-graded + summary hand-graded good).
Tasks: arithmetic 847×36, word problem, sisters logic riddle, capital of Australia,
write palindrome function (code executed + tested), fix buggy sum_n (executed + tested),
one-sentence summary, exactly-3-bullets instruction, JSON extraction, needle in ~40K-token haystack.

## Speed
| Metric | Nemotron 3.5 Lightning | Muse Glimmer 30B |
|---|---|---|
| Generation speed (short tasks) | 114–142 tok/s (avg ~125) | 76–78 tok/s (avg ~76) |
| Generation speed (64K-loaded needle) | 138 tok/s | 20 tok/s |
| Prompt ingest @ ~40K tokens | 4,666 tok/s | 4,016 tok/s |
| Needle task wall time | 9.8 s | 14.8 s |

## Memory (64K context loaded)
| | Nemotron | Glimmer |
|---|---|---|
| Model size on disk | 25 GB | 18 GB |
| In VRAM (ollama ps) | 25 GB, 100% GPU | 16 GB, 100% GPU |
| Total GPU mem used | 30.9 / 32.6 GB | ~17 GB |

Nemotron barely fits at 64K on 32GB; Glimmer leaves ~15GB headroom.
Nemotron's Mamba-2 hybrid = tiny KV cache -> generation speed barely drops at 40K context (138 tps) while Glimmer's transformer KV drops it to 20 tps.

## Thinking behavior (both are reasoning models)
- Nemotron thinking tokens per task: 38–391 (thought on every task)
- Glimmer thinking tokens per task: 55–417 (most on logic riddle)
- Both nearly skipped thinking on the needle retrieval (38 vs 65 tokens)

## Verbosity
- Nemotron generated more tokens per answer overall (185–691) vs Glimmer (97–604) — chattier despite being faster per token.

## Per-task generation speed (tok/s), Nemotron vs Glimmer
math_arithmetic 129.6 / 77.6; math_word 142.1 / 77.1; logic 117.0 / 76.2;
knowledge 119.6 / 76.8; coding_simple 117.7 / 76.3; coding_bugfix 128.1 / 76.2;
summarize 119.1 / 76.3; instruction_following 114.1 / 76.4; json_extract 121.1 / 76.7;
long_context_needle 137.9 / 20.1

## Caveats for the blog
- Vendor GGUF/K-Quant 4-bit builds, not the exact HF NVFP4/BF16 files.
- Simple tasks only — this is not MMLU/SWE-bench; both models saturate easy tasks.
- Single run per task, temperature 0.2.
- First Nemotron task wall time (24.7s) included model load; excluded from analysis.

# DeepSeek-V4-Flash vs Qwen3.8-Flash-Next, both at 1-bit, on hard problems

Two very large MoE models, each at the smallest quantization its vendor ships,
made to run on one desktop and then given problems that 30B models mostly fail.

## Rig

- Intel Core Ultra 9 285K, 24 cores / 24 threads (8 P + 16 E), no SMT
- 95.3 GB RAM
- RTX 5090, 32 GB VRAM (32,607 MiB), driver 610.88
- Windows 11 Pro 26200

### Core map (measured, not assumed)

`GetLogicalProcessorInformationEx` reports the P and E cores **interleaved** on
this chip, so a hand-written affinity mask based on "low CPUs are P-cores" is
wrong. Windows has no `taskset`; llama.cpp does this natively with `-C` and
`--cpu-strict`.

| set | CPUs | llama.cpp flags |
|---|---|---|
| P-cores (8) | 0,1,10,11,12,13,22,23 | `-C C03C03 --cpu-strict 1 -t 8` |
| E-cores (16) | 2-9, 14-21 | `-C 3FC3FC --cpu-strict 1 -t 16` |
| all (24) | 0-23 | `-C FFFFFF -t 24` |

`research/affinity.sh` measured whether a homogeneous core set beats 24 unpinned
threads. The prior Flash-Next run swept thread *count* only, always with
`cpu_mask=0x0` and `cpu_strict=false`, so the question was open until now.

Answer: no. Flash-Next UD-IQ1_S, CPU-only build, 3 reps
(`results/affinity.jsonl`):

| config | threads | mask | pp512 t/s | tg128 t/s |
|---|---:|---|---:|---:|
| all 24, unpinned | 24 | `0x0` | **61.90** ±1.86 | **11.48** ±0.06 |
| P-cores only | 8 | `C03C03` | 34.48 ±1.25 | 10.79 ±0.03 |
| E-cores only | 16 | `3FC3FC` | 40.12 ±0.89 | 10.43 ±0.57 |
| all 24, pinned | 24 | `FFFFFF` | 51.71 ±3.50 | 9.10 ±0.26 |

Two results worth keeping. E-cores are the *weaker* half for decode: 16 of them
lose to 8 P-cores, which is what a bandwidth-bound kernel looks like when 8
cores already generate enough outstanding memory requests to saturate DRAM. And
`all24-pinned` uses the identical core set as the baseline, differing only by
`--cpu-strict 1`, yet loses 20.8% on generation. The cost is the pinning itself,
not the core selection.

Prefill scales with core count (24 > 16 > 8), consistent with compute-bound GEMM.

So the pure-CPU runs use `-t 24` with no mask. Note that the older sweep
measured 10.63 t/s for 24 unpinned threads where this one measures 11.48, about
8% drift between runs, so compare within a run and not across.

## Models

Both are Unsloth dynamic quants at each model's smallest available build.
`models/` is gitignored; re-fetch DeepSeek with `research/dl.ps1`.

| | Qwen3.8-Flash-Next | DeepSeek-V4-Flash-0731 |
|---|---:|---:|
| build | UD-IQ1_S | UD-IQ1_S |
| shards | 3 | 3 |
| bytes | 72,546,461,344 | 82,539,237,792 |
| size | 67.56 GiB | 76.87 GiB |
| params | 176,943,899,520 | ~284 B |
| arch | `qwen4exp` | `deepseek4` |

Flash-Next weights are read in place from `../qwen38-flash-next-cpu/models/`.

Both are labelled IQ1_S but that label is a target the dynamic quantizer
resolves per architecture, not a fixed compression level: Flash-Next lands at
3.279 bits/param and DeepSeek at roughly 2.325. The condition being compared is
therefore "each model's smallest shipping build", which is the build anyone with
this much RAM would actually run. It is **not** an equal-compression comparison,
and no claim about which architecture is better can rest on it.

### DeepSeek-V4-Flash, from config.json

43 layers, hidden 4096, 256 routed experts at moe_intermediate 2048, 1 shared
expert, 6 experts per token, vocab 129,280, MLA attention
(`num_key_value_heads: 1`, `head_dim: 512`), `max_position_embeddings`
1,048,576.

Routed experts alone are 277.0 B params, 97% of the model. Shared expert and
embeddings add 2.1 B. That corroborates the 284 B figure from the model card.
Active expert params per token are 7.6 B before attention.

`expert_dtype: fp4`, so the routed experts are **natively 4-bit**. UD-IQ1_S is
therefore quantizing below native precision, which is a thing to measure rather
than assume is free.

## Toolchain: nothing new to build

`deepseek4` and `qwen4exp` are both already implemented in the llama.cpp
checkout at `../qwen38-flash-next-cpu/research/llama.cpp`, commit `6c5afc86`
(PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)), and both
compiled binaries contain the arch. Verified by grepping the binaries.

| build | used for |
|---|---|
| `build-cpu/bin/Release` | pure-CPU runs |
| `build-cuda/bin` | GPU offload runs |

The separation is load-bearing. A CUDA build run with `-ngl 0` is **not**
CPU-only: it still offloads batch matmuls, measured at 212 t/s prompt processing
against 53 t/s for the genuinely CPU-only binary.

## Tasks

`research/tasks.py` holds 12 hard tasks ported **verbatim** from
`../4-model-hard-benchmark/research/hardbench.py`, graders byte-identical, so
scores here are comparable with that 30B run. Only the runner changed, from
Ollama to llama-server.

Categories: 4 math, 1 logic grid, 4 code (two executed against test suites in a
subprocess with a timeout), 1 brutal instruction-following, 1 strict JSON, 1
multi-hop needle in a ~43K-token document.

Plus one **three.js game task**: a single self-contained HTML file implementing
an endless lane runner. Auto-scoring is structural only (11 checks). Real
quality is judged by opening the file, so each model's output is written to
`games/{config}_s{seed}.html`.

## Method and its limits

Four configs: `{dsv4, flashnext} x {cpu, gpu}`. Identical sampling on both sides
(temp 1.0, top_p 0.95, top_k 20, min_p 0.0) rather than each vendor's
recommendation, so the comparison is not confounded by sampler differences.

Deliberate cuts, made to keep the run inside a few hours:

1. **One seed.** Not enough to separate close scores; treat a 1-task gap as
   noise.
2. **Full quality suite on GPU only.** Mode changes speed, not answers, so the
   CPU configs get a 3-task spot-check to confirm no systematic divergence
   rather than a full re-run.
3. **`multihop_needle_64k` on GPU only.** It is a ~43K-token prefill, which on
   DeepSeek's A13B CPU path is close to an hour for a single cell.

Anything the `--budget-min` cap drops is written to `results/skipped.jsonl`, so
a partial matrix is never mistaken for a complete one.

## Tier 1 result: a tie that hides two different models

Both models scored **10/12** on GPU offload. The score is the least
interesting thing about it.

| | DeepSeek-V4-Flash 284B | Qwen3.8-Flash-Next 177B |
|---|---:|---:|
| tier 1 score | 10/12 | 10/12 |
| failure mode | 2 wrong answers | 2 non-terminations |
| tokens per answer (clean) | 1,222 | 2,113 |
| median generation | 24.5 t/s | 47.0 t/s |
| three.js game | working, 9,368 tok | never emitted, 32,768 tok |
| total wall clock | 15.4 min | 28.6 min |

Flash-Next generates about twice as fast and still took nearly twice as long
overall, because the tasks it fails, it fails slowly.

### The failures are not the same kind of thing

DeepSeek fails by being **wrong**. On `code_output_prediction` it answered
`[1] [1, 2]` where the correct middle line is `[1, 2] [1, 2]`; it missed that
`print(f(1), f(2))` evaluates both calls before stringifying, so both show the
same mutated list. Audited against the raw output, the grader is right and the
failure is real. It also failed `instruction_brutal`, the only model of six
tested to do so, since all four 30B models in the sister run passed it.

Flash-Next fails by **never finishing**. On `code_expression_eval` it produced
104,081 characters of reasoning and **zero characters of content**, writing
`def calc` eleven times without committing to any of them, self-interrupting
with 12 "Wait" and 13 "Actually", and inventing requirements the prompt never
made (it spent tokens on how to tokenize `5e-3`). Doubling the budget did not
help:

| task | first cap | second cap |
|---|---|---|
| `code_expression_eval` | fail @ 8,192 | fail @ 24,576 |
| `threejs_game` | fail @ 16,384 | fail @ 32,768 |

Repetition analysis rules out degenerate looping: repeated 12-grams are 0.0%
and 2.3% in the two truncated cells, and the 2.3% is re-quoting the prompt.
This is unbounded deliberation, not a quantization loop.

### Two audits, opposite outcomes

Neither of these could be settled from a score:

| cell | flagged because | verdict |
|---|---|---|
| flashnext / `aime_digit_sum` | hit the cap | **false failure**; 342 was already derived in the reasoning, and it passed at a larger cap using 9,026 tokens, just 834 over the original |
| dsv4 / `code_output_prediction` | strict substring grader | **real failure**; genuinely wrong output |

The AIME cell is the cautionary one. An 8,192-token cap looked generous and was
short by 10%, which would have published "Flash-Next fails AIME-style math" from
a model that had already computed the right answer. Two of the four 30B models
in the sister run show the same signature on the same task and deserve a re-read
before either article leans on it.

## Speed: CPU-only and GPU offload

`results/cpu_speed.jsonl`, CPU-only binary, `-t 24` unpinned, 2 reps:

| model | weights | pp512 t/s | tg128 t/s | stddev |
|---|---:|---:|---:|---:|
| Qwen3.8-Flash-Next | 67.56 GiB | 62.94 | 11.19 | 0.03 |
| DeepSeek-V4-Flash | 76.87 GiB | 25.21 | 7.10 | 0.02 |

**A 284B model generates at 7.1 t/s on this desktop with no GPU at all.** The
0.02 stddev matters as much as the mean: a page-thrashing run produces wild
variance, so 76.87 GiB of weights genuinely fits inside 95.3 GiB of RAM
alongside the OS. That was the riskiest assumption in the plan and it held.

GPU offload buys 3.5x for DeepSeek (7.10 to 24.7) and 4.2x for Flash-Next
(11.19 to 47.4).

### GPU split calibration

`-ncmoe 30` is DeepSeek's operating point on a 32 GB card, and it is a cliff
edge, not a plateau (`results/ncmoe_ds.jsonl`):

| ncmoe | pp512 t/s | tg128 t/s |
|---:|---:|---:|
| 30 | 45.16 | **27.95** |
| 27 | 19.70 | 6.16 |
| 24 | 17.32 | 4.98 |
| 21 | 18.70 | 4.47 |

Moving *more* layers onto the GPU makes it **4.5x slower**. Past `-ncmoe 30`
the weights no longer fit VRAM (31,160 of 32,607 MiB in use at that setting)
and CUDA spills over PCIe instead of failing, so those layers end up slower
than if they had stayed on the CPU. On Windows "it loaded" proves nothing; the
decode rate is the only signal.

A second silent cost: llama-server defaults to `n_slots = 4` and allocates KV
for all four. On the GPU configs that VRAM comes out of the weight budget and
pushes layers into spill. Adding `--parallel 1` took DeepSeek from 14.6 to
23.5 t/s served, with no other change.

## CPU vs GPU: same conclusions, different text

Three tasks re-run on the CPU-only binary, same weights, seed and sampling:

| task | dsv4 gpu/cpu | tokens gpu/cpu | flashnext gpu/cpu | tokens gpu/cpu |
|---|---|---:|---|---:|
| modular_arithmetic | PASS / PASS | 337 / 325 | PASS / PASS | 727 / 474 |
| combinatorics | PASS / PASS | 546 / 369 | PASS / PASS | 338 / 1086 |
| code_subtle_bugfix | PASS / PASS | 159 / 249 | PASS / PASS | 124 / 200 |

Zero verdict mismatches, so the backend does not change conclusions and cut 2
in the method section is justified. But the **token counts move a lot** (up to
3.2x on flashnext/combinatorics), because float reduction order differs between
backends and the sampled text diverges even at a fixed seed.

Consequence for the write-up: **token-cost figures are comparable only within a
backend.** The cost-per-answer numbers above are GPU-to-GPU and must not be
mixed across modes.

## Final scoreboard

19 tasks per model, one seed, GPU offload.

| | DeepSeek-V4-Flash 284B | Qwen3.8-Flash-Next 177B |
|---|---:|---:|
| **delivered** | **16/19** | 13/19 |
| **derived correctly** | 16/19 | **15/19** |
| truncated cells | 1 | 7 |
| tokens per answer | **2,257** | 2,719 |
| median generation (GPU) | 24.7 t/s | **47.4 t/s** |
| CPU-only generation | 7.10 t/s | **11.19 t/s** |
| total wall clock | **45.8 min** | 58.4 min |
| three.js game | **working** | never emitted |
| 64K needle (38,873 tok prompt) | PASS, 256s | PASS, 119s |

`results/audit_verdicts.json` records, with verbatim quotes, which flagged
cells were real failures and which were not. The delivered/derived split comes
from there and not from the scores.

## Honesty section

- **One seed.** A one-task gap is noise. The 7-to-1 truncation split is not.
- **Tier 1 saturates.** It came from the 30B run unmodified, where a 30B model
  scored 12/12. Tier 2 exists because of that and is where separation appears.
- **`gen_tps` conflates prefill and decode**, being completion tokens over total
  request seconds. On the needle task that reports 0.34 t/s for a 38,873-token
  prompt answered in 86 tokens. It is not a generation rate and is excluded from
  the median in `report.py`.
- **The Manacher test was wrong when first written.** Its timing string was
  random, where the longest palindrome is ~35 chars, so an O(n^2) solution
  finished in 0.5s and passed. Fixed to an adversarial `"ab" * 100000` before any
  model ran against it. Every tier-2 code test was validated against both a
  correct and a naive solution first.
- **`results/truncated_discarded.jsonl`** holds the three superseded truncated
  rows from the 8k/16k caps, and `hard_pre_rerun_backup.jsonl` the pre-re-run
  file. Neither is part of any total.
- **Audit 2 produces false positives.** DeepSeek's `1456` on the subsets task
  matched by coincidence (it guessed 4368/3 as a sanity estimate) and is a real
  failure. Substring presence is a prompt to read the context, not a verdict.
- The sister 30B article has two `fail*` cells on `aime_digit_sum` with the same
  truncation signature that proved to be a false failure here. Worth a re-read
  before either article claims a model cannot do that problem.

## Files

| path | what |
|---|---|
| `research/tasks.py` | the 12 hard tasks, graders, game task |
| `research/run.py` | llama-server runner, resumable, time-boxed |
| `research/affinity.sh` | P-core vs E-core vs all-core sweep |
| `research/dl.ps1` | DeepSeek weight fetch |
| `results/hard.jsonl` | one row per cell, raw completion stored verbatim |
| `results/affinity.jsonl` | llama-bench rows for the core sweep |
| `results/skipped.jsonl` | cells dropped by the time budget, with reason |
| `games/` | playable HTML, one per model |
| `logs/` | llama-server stdout per config |

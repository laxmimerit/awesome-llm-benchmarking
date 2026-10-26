# Qwen3.8-Flash-Next on limited hardware

Can a 180B model that officially needs 4x GB300 be run on one desktop? This
folder measures three answers: pure CPU, partial GPU offload, and what the
quantization costs in output quality.

## Rig

- Intel Core Ultra 9 285K, 24 cores / 24 threads (8 P + 16 E), no SMT
- 95.3 GB RAM
- RTX 5090, 32 GB VRAM, driver 610.88, CUDA 13.1
- Windows 11 Pro 26200

## Why this needed a source build

`qwen4exp` is not in llama.cpp master. Support lives in open PR
[#27742](https://github.com/ggml-org/llama.cpp/pull/27742), checked out here at
commit `6c5afc86` (2026-08-27). Two builds are kept side by side and they are
not interchangeable:

| build | `--list-devices` | used for |
|---|---|---|
| `research/build-cpu`  | `(none)` | the pure-CPU numbers |
| `research/build-cuda` | RTX 5090 | the offload ladder, perplexity, quality |

The separation is deliberate. A CUDA build run with `-ngl 0` is **not** CPU-only:
it still offloads batch matmuls, and measured 212 t/s prompt processing against
53 t/s for the genuinely CPU-only binary. Any "CPU baseline" taken with `-ngl 0`
on a CUDA build is really a GPU measurement.

On Windows, CMake picks the VS 2026 BuildTools, which CUDA 13.1 has no MSBuild
integration for. `research/build_cuda.bat` sidesteps that with Ninja + VS 2022.

## Models

Weights are Unsloth dynamic quants. `models/` is gitignored; re-fetch with
`research/dl.ps1 -Quant UD-IQ1_S -Shards 3`.

| file | bytes | note |
|---|---:|---|
| `UD-IQ1_S`   | 72,546,461,344 | smallest build that exists |
| `UD-Q2_K_XL` | 78,869,128,864 | largest that fits 96 GB RAM comfortably |

4-bit was deliberately **not** used: `UD-IQ4_XS` is 93.7 GB and `UD-Q4_K_XL` is
111.3 GB, both at or past this box's RAM ceiling.

The Qwen3.8-27B comparison models are read in place from
`../qwen38-unsloth-dynamic3/models/` and are the same Unsloth quant recipes, so
27B-vs-Flash-Next is a controlled comparison rather than a cross-vendor one.

## The number that shapes everything

In the **1-bit** build, `per_layer_token_embd` - the n-gram lookup table - is
**26.82 GiB of the 67.55 GiB total (39.7%)** and is stored at **IQ4_NL, i.e. 4-bit**.
It is registered `GGML_OP_GET_ROWS`: pure gather, zero matmul. Unsloth's dynamic
quant refuses to crush it, which is why no smaller build than 72.5 GB exists.

| component | GiB | % | tensors |
|---|---:|---:|---:|
| MoE expert banks | 37.11 | 54.9 | 144 |
| PLE n-gram table | 26.82 | 39.7 | 1 |
| attention / QSA | 1.71 | 2.5 | 384 |
| everything else | 1.91 | 2.8 | 695 |

Only ~3.6 GiB is dense weight touched every token. That is what makes offload
work on a 32 GB card, and why `-ot per_layer_token_embd=CPU` is the right call:
a zero-FLOP gather table should never take VRAM from expert banks.

## Harness

| script | purpose | output |
|---|---|---|
| `research/dl.ps1` | parallel shard download via curl | `models/` |
| `research/bench.py` | speed sweeps (`cpu`, `gpu`, `longctx`, `longctx_gpu`) | `results/{stage}.jsonl` |
| `research/ppl.py` | wikitext-2 perplexity | `results/ppl.jsonl` |
| `research/quality.py` | 12-problem suite via llama-server | `results/quality.jsonl` |
| `research/problems.json` | the problem set and its checkers | - |

Every JSONL row carries the exact command that produced it.

## Rules followed here

- Perplexity is compared across models only because the tokenizers were verified
  byte-identical (vocab 248,320, gpt2 BPE, `pre=qwen35`, same BOS/EOS). Across
  different tokenizers it would be meaningless.
- Ground truth in `problems.json` was independently recomputed before use. Three
  of the originally written answers were wrong and were corrected.
- Quality is graded on `content` only; `reasoning_content` is stored separately
  so thinking text cannot accidentally satisfy a checker.
- Responses record `finish_reason`. A thinking model that exhausts its token
  budget mid-reasoning returns empty content, which looks identical to a wrong
  answer unless truncation is tracked. The first pilot at `max_tokens 4096` lost
  3 of 12 problems this way; the real runs use 16384.
- `results/*_discard.jsonl` are pilot runs. They are **not** report data.

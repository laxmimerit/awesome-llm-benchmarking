# Lecture 186: Qwen 3.8 27B Speed Settings and KV Cache Explained

This project investigates **45 distinct `llama.cpp` runtime configurations** for **Qwen 3.8 27B** on an **NVIDIA RTX 5090 (32GB VRAM)**. We empirically measure the multi-token prediction (MTP) draft head ladder, KV cache quantization (`f16`, `q8_0`, `q4_0`), context size scaling, and reasoning effort flags to determine which tuning knobs actually improve inference speed and which are myths.

![Decode speed across the MTP draft ladder](figures/chart-mtp-ladder.png)

*The whole project in one chart: speculative decoding is worth 1.81x, but only
up to a draft depth of 3, after which deeper drafts cost more than they save.*

---

## Hardware & Environment

- **GPU**: NVIDIA GeForce RTX 5090 (32 GB GDDR7 VRAM, 600W power limit)
- **CPU**: Intel Core Ultra 9 285K (24 cores / 24 threads)
- **RAM**: 96 GB DDR5
- **OS**: Windows 11 Pro
- **Runtimes**: `llama.cpp` (build b10448) with CUDA 13.1 backend, Ollama 0.32.13
- **Model**: Qwen 3.8 27B (`Q4_K_M` Ollama GGUF, containing 64 transformer layers + 1 built-in MTP layer at index 64)

---

## Verified Truth vs. Viral Claims

| Claim / Flag | Empirical Verdict | Measured Data |
|---|---|---|
| `--draft-max` gives free speedup | **TRUE** | Up to **1.81x faster** decoding without quality degradation |
| Model context is 1M tokens | **FALSE** | Model metadata specifies **262,144 tokens** (`qwen35.context_length`) |
| $n=2$ is the optimal draft depth | **PARTLY** | $n=3$ wins on `f16` KV cache (1.81x vs 1.70x); with `q4_0` the two tie on speed (136.75 vs 136.72 tok/s) and $n=2$ wins on memory, costing 150 MB less |
| $n=4$ crashes or emits garbage | **FALSE** | Runs stably with 100% answer accuracy; acceptance simply drops to 59.2% |
| Use `q8_0` KV cache | **FALSE** | `q4_0` beats `q8_0` on both memory savings AND decode speed |
| Reasoning defaults to high; set `medium` | **TRUE** | `--reasoning-effort medium` eliminates unnecessary thinking latency |

---

## The Speculative Decoding MTP Ladder (32K Context, f16 KV Cache)

| Setting | Decode tok/s | Speedup | Acceptance Rate | Tokens/Step | VRAM (GB) | Energy (tok/Wh) | Quality |
|---|---|---|---|---|---|---|---|
| **Base (MTP Off)** | 73.6 | 1.00x | — | 1.00 | 17.54 | 485 | 100% |
| **Draft $n=1$** | 104.8 | 1.42x | 86.0% | 1.88 | 18.21 | 691 | 100% |
| **Draft $n=2$** | 125.5 | 1.70x | 76.6% | 2.56 | 18.36 | 768 | 100% |
| **Draft $n=3$ (Peak)** | **133.6** | **1.81x** | 67.4% | 3.11 | 18.50 | **812** | 100% |
| **Draft $n=4$** | 119.5 | 1.62x | 59.2% | 3.56 | 18.65 | 730 | 100% |
| **Draft $n=5$** | 108.8 | 1.48x | 52.0% | 3.90 | 18.79 | 684 | 100% |

![Draft acceptance rate against draft depth](figures/chart-mtp-acceptance.png)

*Acceptance falls steadily as the draft gets deeper, from 86.0% at n=1 to 52.0%
at n=5. Peak throughput sits where accepted tokens per step stop outrunning the
cost of the rejected ones.*

---

## KV Cache Quantization Comparison

Evaluating KV cache formats at 32K context:

| KV Cache Type | VRAM, no MTP | VRAM, MTP $n=2$ | Decode, no MTP | Decode, MTP $n=2$ | Quality Score |
|---|---|---|---|---|---|
| **`f16` (Default)** | 17.54 GB | 18.36 GB | 73.6 tok/s | 125.5 tok/s | 100% |
| **`q8_0`** | 16.69 GB | 17.50 GB | 71.8 tok/s | 128.3 tok/s | 100% |
| **`q4_0` (Winner)** | **16.19 GB** | **17.00 GB** | **77.1 tok/s** | **136.7 tok/s** | 100% |

Against `f16`, `q4_0` saves 1.35 GB without a draft head and 1.36 GB with
$n=2$. Draft depth 2 is shown because it is the only depth measured for all
three cache types. The best single result in the whole sweep is `q4_0` at
$n=3$: **136.8 tok/s** at 17.15 GB, which beats `f16` at its own best depth
(133.6 tok/s at 18.50 GB) while using 1.35 GB less.

> **Recommendation**: `--cache-type-k q4_0 --cache-type-v q4_0` takes **1.35 GB**
> off the process footprint at 32K context with **zero loss in answer quality**,
> including needle retrieval at three depths, and is the fastest of the three
> cache types at every draft depth measured. `q8_0` is the one to skip: it saves
> less memory and, without MTP, decodes slower than plain `f16`.

![KV cache VRAM by cache type](figures/chart-kv-vram.png)

*Peak VRAM for the three cache types, all at MTP $n=2$, which is the middle
column of the table above.*

![Answer quality by cache type](figures/chart-kv-quality.png)

*The memory saving is real and the quality cost, on this problem set, is not
there to find. That is what makes `q4_0` the default worth changing.*

---

## Recommended `llama-server` Launch Command

```bash
llama-server \
  -m models/qwen3.8-27b-q4_k_m.gguf \
  -c 32768 \
  -fa 1 \
  --draft-max 3 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --reasoning-effort medium \
  --port 8080
```

---

## Context scaling

![Prompt processing speed across context sizes](figures/chart-ctx-speed.png)

![VRAM across context sizes](figures/chart-ctx-vram.png)

![Full prompt latency across context sizes](figures/chart-ctx-fullprompt-latency.png)

## Reasoning effort

![Time to first token by reasoning effort](figures/chart-reasoning-latency.png)

![Answer quality by reasoning effort](figures/chart-reasoning-quality.png)

![Thinking tokens by reasoning effort](figures/chart-reasoning-tokens.png)

![Decode speed while thinking](figures/chart-thinking-speed.png)

![Runtime comparison](figures/chart-runtime.png)

All twelve charts live in [`figures/`](figures/).

---

## Reproduction

See [`research/`](research/) for benchmark driver scripts and measurement logs. See [`FINDINGS.md`](FINDINGS.md) for the complete raw telemetry report across all 45 configurations.

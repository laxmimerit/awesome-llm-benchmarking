# Lecture 189: Qwen 3.8 Flash Next (Qwen 4) Architecture Teardown

This project provides an architectural teardown of **Qwen3.8-Flash-Next** (officially tagged as `qwen4_exp`, the preview architecture for **Qwen 4**) by performing a block census and tensor diff against **Qwen 3.8 27B** using their official Hugging Face configurations and safetensors index maps.

![The four changes from Qwen 3.8 27B to Flash Next](figures/social-hf-four-changes.png)

---

## Model Scale: 180B Total Weights, 6B Active per Token

| Metric | Qwen 3.8 27B (Dense) | Qwen3.8-Flash-Next (`qwen4_exp`) | Architectural Impact |
|---|---|---|---|
| **`model_type`** | `qwen3_5` | `qwen4_exp` | Previews Qwen 4 architecture |
| **Total Parameters** | 27.3 Billion | **~180 Billion** | 6.5x larger parameter memory capacity |
| **Active Parameters / Token** | **27.3 Billion** | **~6.0 Billion** | 4.5x cheaper compute per generated token |
| **Transformer Blocks** | 64 blocks | 48 blocks | Shallower layer stack |
| **Hidden Dimension ($d_{\text{model}}$)** | 5,120 | 2,560 | Half the width of 27B |
| **Feed-Forward Layers (FFN)** | 1 dense FFN ($17,408$) | **512 routed experts + 1 shared** | Fine-grained MoE |
| **MoE Routing** | None (Dense) | Top-8 experts active per token | Extreme specialization per domain |
| **Layer Interleaving** | 3 DeltaNet : 1 Attention | 3 DeltaNet : 1 Attention | Preserved 3:1 linear-to-attention ratio |
| **Full Attention Heads** | 24 Q, 4 KV (dim 256) | 24 Q, 4 KV (dim 256) | Preserved attention geometry |
| **Native Context Length** | 262,144 tokens | 262,144 tokens | Identical ultra-long context |
| **Multi-Token Prediction (MTP)** | 1 draft head | 1 draft head | Built-in speculative decoding |

![Total against active parameters](figures/chart-params.png)

*Counted from the index maps, not read off the model card: 180B stored against
roughly 6B touched per token.*

---

## The Four Key Architectural Shifts

### 1. The 512 Fine-Grained Routed Experts
Instead of a single dense FFN ($d_{\text{ffn}} = 17,408$), Flash-Next replaces the MLP with **512 fine-grained routed experts** plus 1 shared expert. For each token, the router selects only the top 8 experts. This keeps active computation down to only **~6B parameters**, enabling blistering inference speeds while storing 180B parameters of knowledge.

![The 512 expert grid](figures/chart-expert-grid.png)

### 2. Shallower, Narrower Hidden Dimension
By reducing $d_{\text{model}}$ from 5,120 to 2,560 and total layers from 64 to 48, communication overhead between tensor-parallel ranks shrinks dramatically, making massive scaling across cluster nodes efficient.

![Block by block stack map](figures/chart-stack-map.png)

*Every block in both stacks, by layer type. The 3 DeltaNet to 1 attention
interleave survives the rebuild.*

### 3. Radical KV Cache Footprint Reduction
With only 12 full attention layers across the 48 blocks (1 every 4 layers) and a narrower key/value projection, KV cache size per token drops by over **70%**, enabling massive batch sizes and 128K+ sequence lengths in memory.

![KV cache per token](figures/chart-kv-per-token.png)

![How much each layer reads from attention](figures/chart-attention-read.png)

### 4. Preservation of Hybrid DeltaNet State
Flash-Next retains the 3:1 linear DeltaNet to full attention interleave pattern. The recurrent DeltaNet layers maintain a fixed-size $O(1)$ memory state for 75% of all layers, completely sidestepping quadratic attention scaling.

---

## What it costs to keep

![Storage footprint](figures/chart-storage.png)

![Memory traffic per generated token](figures/chart-token-traffic.png)

![Vendor reported benchmark delta](figures/chart-benchmark-delta.png)

![Speedup from the MTP draft head](figures/chart-mtp-speedup.png)

![What the parameter count costs to run](figures/social-linkedin-cost.png)

All charts live in [`figures/`](figures/) and are drawn straight from the two
config files and their index maps.

---

## Data & Scripts

- [`data/qwen3.8-27b.config.json`](data/qwen3.8-27b.config.json): Official Qwen 3.8 27B model configuration.
- [`data/qwen3.8-27b.index.json`](data/qwen3.8-27b.index.json): Safetensors tensor name-to-shard mapping.
- [`data/qwen3.8-flash-next.config.json`](data/qwen3.8-flash-next.config.json): Official Qwen3.8-Flash-Next configuration.
- [`data/qwen3.8-flash-next.index.json`](data/qwen3.8-flash-next.index.json): Safetensors tensor name-to-shard mapping.
- [`figures/make_figures.py`](figures/make_figures.py): Generates all architectural comparison diagrams directly from the config and index files.

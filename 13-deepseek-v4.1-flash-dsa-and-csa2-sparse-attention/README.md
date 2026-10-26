# Lecture 196: DeepSeek V4.1 Flash Architecture Explained - DSA and CSA2 Sparse Attention

This project investigates the architectural innovations in **DeepSeek V4.1 Flash**, focusing on its **Dynamic Sparse Attention (DSA)** and **Compressed Sparse Attention 2 (CSA2)** mechanisms, fine-grained **FP8 KV cache quantization**, and the Hopper **Distributed Shared Memory (DSM) crossover kernel**.

---

## Architectural Breakthrough: DSA & CSA2

Standard dense attention requires calculating attention across all $L$ tokens in context, incurring $O(L^2)$ computation during prefill and $O(L)$ per decoded token. For ultra-long context ($128\text{K}+$ tokens), KV cache memory bandwidth becomes the defining bottleneck.

DeepSeek addresses this with two coordinated sparse mechanisms:

1. **Dynamic Sparse Attention (DSA)**: Dynamically selects the top-$k$ most relevant tokens (e.g., $k=512$ or $2048$) per query token at runtime, reducing compute and memory access by up to 96%.
2. **Compressed Sparse Attention 2 (CSA2)**: Compresses key and value projections into compact latent representations using Multi-Head Latent Attention (MLA), paired with tile-level FP8 quantization.

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Sequence                         │
│                    (Up to 128K Tokens)                      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
   Dense Prefill Compression        Dynamic Token Router
    - RoPE split (64 bf16)           - Evaluates token relevance
    - Latent KV (512 fp8)            - Selects active top-k (512)
                │                             │
                └──────────────┬──────────────┘
                               ▼
               ┌───────────────────────────────┐
               │  Hopper Crossover MMA Kernel  │
               │  - DSM (Distributed Shmem)    │
               │  - 2-CTA cluster per SM       │
               │  - 410+ TFLOPS FP8 throughput │
               └───────────────────────────────┘
```

---

## Key Technical Specifications

From [`sources/v32exp_config.json`](sources/v32exp_config.json) and [`sources/deepdive.md`](sources/deepdive.md):

| Feature | Value / Specification |
|---|---|
| **Context Length** | 128,000 tokens |
| **Attention Architecture** | Multi-Head Latent Attention (MLA) with MQA decoding |
| **Query Heads** | 128 heads |
| **Key/Value Heads** | 1 shared key head (`head_dim_k = 576`, `head_dim_v = 512`) |
| **KV Cache per Token** | 656 bytes (512 `float8_e4m3` + 4 `float32` scales + 64 unquantized `bfloat16` RoPE) |
| **Memory Reduction** | >75% reduction compared to standard 16-bit dense KV cache |
| **Sparse Selection ($k$)** | 512 / 2048 active tokens selected dynamically |
| **Hopper Kernel Throughput** | 410 TFLOPS (at $k=2048$) to 460 TFLOPS (at $k=32768$) on H800 SXM5 |

---

## The Hopper DSM Crossover Kernel Innovation

On NVIDIA Hopper GPUs (H100/H800), CUDA dequantization instructions (`float8` → `half` → `float32` → `bfloat16`) consume ~50 clock cycles per token, while Tensor Core MMA operations take only ~34 cycles—making standard dequantization memory- and CUDA core-bound.

DeepSeek's **Crossover Technique** solves this:
1. Two CTAs (CUDA Thread Blocks) launch in a **size-2 cluster** mapped to the same query token (each processing 64 heads).
2. Each CTA loads and dequantizes only **half** of the KV cache from global memory.
3. CTAs exchange the dequantized values using **Hopper Distributed Shared Memory (DSM)** with asynchronous copy instructions (`st.async`) and cluster transaction barriers.
4. Tensor Cores execute MMA at full occupancy without stalling for CUDA core dequantization.

---

## Repository Contents

- [`dsa-visual-story.html`](dsa-visual-story.html): Interactive visual walkthrough explaining DSA indexing, token routing, and attention weight sparsity.
- [`sources/deepdive.md`](sources/deepdive.md): Full engineering deep dive into the Flash MLA FP8 Hopper kernel.
- [`sources/model.py`](sources/model.py): PyTorch reference implementation of the sparse attention architecture.
- [`sources/v32exp_config.json`](sources/v32exp_config.json): Architecture configuration JSON.
- [`sources/flashmla_readme.md`](sources/flashmla_readme.md): Build and benchmark instructions for FlashMLA.

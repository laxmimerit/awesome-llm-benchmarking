# Lecture 194: Qwen 3.8 27B Token Generation Speed Tested with MacBook Pro M5 Max

This project benchmarks **Qwen 3.8 27B** on Apple Silicon using a **MacBook Pro M5 Max (36GB Unified Memory)** and compares its decode throughput, Multi-Token Prediction (MTP) draft head scaling, and memory characteristics against the **NVIDIA RTX 5090** desktop reference rig.

---

## Hardware & Runtime Specifications

| Parameter | MacBook Pro M5 Max | RTX 5090 Reference Rig |
|---|---|---|
| **Processor** | Apple M5 Max (32 GPU cores) | Intel Core Ultra 9 285K (24 cores) |
| **GPU / Memory** | 36 GB Unified LPDDR5 (28.1 GB addressable to GPU) | NVIDIA RTX 5090 (32 GB GDDR7) |
| **OS** | macOS 26.6.2 | Windows 11 Pro |
| **Runtime** | `llama.cpp` (b10793) with Metal backend | `llama.cpp` (b10448) with CUDA 13.1 backend |
| **Model** | Qwen 3.8 27B (`Q4_K_M`, 15.4 GB weights) | Qwen 3.8 27B (`Q4_K_M`, 15.4 GB weights) |
| **Context & Sampling** | 32K context (`-c 32768`), greedy (`temp=0`), 256 token cap | 32K context (`-c 32768`), greedy (`temp=0`), 256 token cap |

---

## Key Experimental Findings

### 1. Unified Memory Allocation
On a 36 GB Apple Silicon Mac, llama.cpp reports:
```
cmn  common_param:   - MTL0    : Apple M5 Max (28753 MiB, 28753 MiB free)
cmn  common_param:   - CPU     : Apple M5 Max (36864 MiB, 36864 MiB free)
```
- GPU can address **28,753 MB** (~28.1 GB).
- Q4_K_M model weights take **15.4 GB**, leaving **12.7 GB** for the KV cache, recurrent state, and scratch buffers.
- Turning Flash Attention (`-fa 1`) on shrinks compute buffer requirements from **1,664 MB** down to **257 MB**, saving critical memory.

### 2. MTP Draft Depth Scaling (The Draft Ladder)
Qwen 3.8 includes a built-in Multi-Token Prediction draft head (layer index 64). Testing draft depths from `n=0` to `n=4`:

| Draft Depth | MacBook Pro M5 Max (Best) | MacBook Acceptance | RTX 5090 Reference | RTX 5090 Acceptance |
|---|---|---|---|---|
| **n=0 (No MTP)** | 18.2 tok/s | — | 74.2 tok/s | — |
| **n=1** | 23.4 tok/s | 88.5% | 98.1 tok/s | 88.9% |
| **n=2** | 27.8 tok/s | 78.4% | 118.6 tok/s | 79.1% |
| **n=3 (Peak)** | **31.1 tok/s** | **71.2%** | **134.2 tok/s** | **71.8%** |
| **n=4** | 30.6 tok/s | 64.9% | 131.0 tok/s | 65.2% |

- **Speedup**: The built-in draft head provides a **1.71x speedup** on the M5 Max (18.2 → 31.1 tok/s) and a **1.81x speedup** on RTX 5090.
- **Cross-Platform Transfer**: Acceptance rate curves transfer almost identically across hardware architectures because acceptance depends on weight statistics and token predictability, not silicon bandwidth.
- **Absolute Speed**: The RTX 5090 decodes ~4.3x faster due to dedicated high-bandwidth GDDR7 memory vs shared system LPDDR5.

### 3. The Laptop "Stall Rate" Phenomenon
Across 3 repeats per configuration on the MacBook:
- 2 out of 3 repeats completed cleanly at ~28–31 tok/s.
- 1 out of 3 repeats stalled down to 0.3–0.7 tok/s due to unified memory pressure and macOS page faults.
- On the RTX 5090 dedicated GPU, zero stalls occurred.
- **Rule for Laptop Benchmarks**: Always report best-of-3 and stall frequency alongside the median.

---

## Visualizations

All generated charts are available in [`figures/`](figures/):
- [`qwen38-mac-draft-ladder.png`](figures/qwen38-mac-draft-ladder.png): Decode throughput scaling across draft depths $n=0 \dots 4$.
- [`qwen38-mac-accept-vs-5090.png`](figures/qwen38-mac-accept-vs-5090.png): Draft acceptance rates of M5 Max vs RTX 5090.
- [`qwen38-mac-repeat-spread.png`](figures/qwen38-mac-repeat-spread.png): Scatter plot illustrating the 1-in-3 stall distribution.
- [`qwen38-mac-flash-attn-memory.png`](figures/qwen38-mac-flash-attn-memory.png): Metal memory allocation with and without Flash Attention.

---

## Reproduction Instructions

### 1. Prerequisites
- macOS running on Apple Silicon (M-series).
- `llama.cpp` built with Metal support (`make` or `cmake -B build -DGGML_METAL=ON`).
- Qwen 3.8 27B GGUF (`qwen3.8-27b-q4_k_m.gguf`).

### 2. Running the Sweep
```bash
cd research
chmod +x benchmark_mac.sh
./benchmark_mac.sh /path/to/qwen3.8-27b-q4_k_m.gguf
```
Or directly via Python:
```bash
python3 benchmark_mac.py --server llama-server --model /path/to/qwen3.8-27b-q4_k_m.gguf --repeats 3
```

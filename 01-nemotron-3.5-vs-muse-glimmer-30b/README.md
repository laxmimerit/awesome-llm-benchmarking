# Lecture 184: Nemotron 3.5 vs Meta's Muse Glimmer 30B on RTX5090

This project provides the complete benchmark harness, raw logs, measurements, and figures comparing **NVIDIA Nemotron 3.5 Lightning** (30B A3B MoE + Mamba-2 hybrid) against **Meta Muse Glimmer 30B** (dense transformer with native vision) on a single **NVIDIA RTX 5090 (32GB VRAM)** rig.

![Generation speed on simple tasks](figures/chart-gen-speed.png)

*Nemotron's 3B active parameters decode roughly 1.6x faster than Muse Glimmer's
29.6B on every short task, while both answer correctly.*

---

## Model Comparison at a Glance

| Specification | NVIDIA Nemotron 3.5 Lightning | Meta Muse Glimmer 30B |
|---|---|---|
| **Developer** | NVIDIA | Meta |
| **Architecture** | Mixture of Experts (MoE) + Mamba-2 Hybrid | Dense Transformer |
| **Total Parameters** | 30 Billion | ~29.6 Billion |
| **Active Parameters** | **3 Billion** (A3B) | **29.6 Billion** (all active) |
| **Max Context** | 1,000,000 tokens | 131,072 tokens |
| **Modalities** | Text only | Multimodal (Text + Vision) |
| **Quantization Evaluated** | 4-bit GGUF (~25 GB on disk) | 4-bit K-Quant (~18 GB on disk) |
| **License** | OpenMDW-1.1 | Apache 2.0 |

---

## Experimental Rig & Methodology

- **Hardware**: NVIDIA GeForce RTX 5090 (32 GB GDDR7 VRAM), Intel Core Ultra 9 285K (24 cores), Windows 11 Pro.
- **Runtime**: Ollama (`num_ctx = 65536`, `temperature = 0.2`).
- **Context Allocation**: Full 64K context loaded into VRAM.
- **Test Battery**:
  - 10 everyday tasks: arithmetic multiplication ($847 \times 36$), word problem, family logic riddle, general knowledge, palindrome checker (executed & unit tested), off-by-one bug fix (executed & unit tested), one-sentence summary, strict 3-bullet instruction following, structured JSON extraction, and a needle in a 40,000-token haystack.

---

## Key Findings

### 1. Accuracy
- **Both models achieved 10/10 (100%)** on the everyday evaluation suite. All math, code executions, JSON schemas, formatting constraints, and the 40K needle retrieval passed without a single failure.

### 2. Decode Throughput
- **Short Tasks**: Nemotron 3.5 Lightning achieved **114–142 tok/s** (average ~125 tok/s). Muse Glimmer 30B achieved **76–78 tok/s** (average ~76 tok/s).
- **Long Context Needle (40K loaded tokens)**:
  - Nemotron's Mamba-2 hybrid state avoids transformer KV cache scaling penalties, maintaining **138 tok/s**.
  - Muse Glimmer's dense attention KV cache overhead caused decode speed to drop to **20 tok/s**.

![Decode speed under a 40K context load](figures/chart-context-speed.png)

*The gap at 40K loaded tokens is the whole argument for the hybrid: one model
barely notices the context, the other loses most of its speed to it.*

### 3. VRAM Consumption & Headroom (at 64K context)
- **Nemotron 3.5**: Model weights take 25 GB. With 64K context loaded, total GPU VRAM utilized reached **30.9 GB / 32 GB** (barely fits on a 32GB card).
- **Muse Glimmer 30B**: Model weights take 18 GB. Total GPU VRAM utilized was **~17 GB**, leaving over **15 GB of headroom**.

![VRAM used and headroom left on a 32 GB card](figures/chart-vram.png)

*Speed is not free. Nemotron spends almost the whole card to get it, which
decides what else you can keep loaded beside it.*

---

## Thinking budget

![Thinking tokens spent per task](figures/chart-thinking.png)

*How much each model thinks before answering, task by task. Two models can
reach the same answer and still spend very different budgets getting there.*

All four charts live in [`figures/`](figures/) and are redrawn from
`research/results.json` by `research/charts.py`.

---

## How to Reproduce

### 1. Pull the Models via Ollama
```bash
ollama pull nemotron-3.5-lightning:latest
ollama pull muse-glimmer:latest
```

### 2. Run the Benchmark Harness
```bash
cd research
python benchmark.py
python hardbench.py
```

### 3. Generate the Charts
```bash
python charts.py
```
Raw outputs will be written to `results.json` and `results-hard.json`.

# Model Card Notes (fetched 2026-08-11)

## NVIDIA Nemotron-3.5-Lightning-30B-A3B-NVFP4
- Source: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
- Architecture: Hybrid Mamba-2 + MoE + select Attention layers
- Params: 30B total, **3B active** (A3B)
- Quantization: NVFP4 PTQ — W4A16 on routed/shared experts, FP8 on mamba in_proj/out_proj and KV cache
- Context: up to **1M tokens**
- Training: 20T+ tokens (web, code, math, synthetic). Pre-train cutoff Sep 2025, post-train May 2026
- Languages: EN, ES, FR, DE, IT, JA + 43 programming languages
- Reported benchmarks: MMLU Pro 81.62, GPQA Diamond 75.57, SWE-bench Verified 52.80, HLE (text) 10.47
- License: OpenMDW-1.1 (commercial OK)
- Intended use: long-running autonomous agents, sub-agent workhorse, efficient local inference
- Hardware: single H100/H200 or DGX Spark (GB10); Blackwell + Ampere supported
- Released: Aug 11, 2026

## Meta Muse-Glimmer-30B
- Source: https://huggingface.co/meta-models/Muse-Glimmer-30B
- Architecture: Dense causal transformer + ~1.8B ViT-G/14 vision encoder (multimodal), 52 layers, alternating local/global attention, SwiGLU FFN
- Params: ~29.6B total (dense — all active)
- Formats: BF16, 4-bit K-Quant-Dynamic, K-Quant-17GB, DFlash drafter (speculative decoding)
- Context: 131,072+ tokens; up to 4,096 visual tokens per image
- Training: multimodal public/3rd-party/Meta data; knowledge cutoff Jan 4, 2026; 100+ languages
- Reported benchmarks: AIME 2026 94.7, GPQA Diamond 83.5, SWE-bench Verified 76.0, SWE-bench Pro 51.2, MCP Atlas 75.5
- License: Apache 2.0
- Intended use: autonomous agentic tasks on consumer hardware, local agents, coding, tool use, multimodal reasoning
- Hardware: 24–32GB consumer GPUs; 4-bit fits 24GB VRAM; 3.1x speedup w/ speculative decoding on RTX 5090
- Released: Aug 2026

## Key structural differences (frame for the blog)
| Dimension | Nemotron 3.5 Lightning | Muse Glimmer 30B |
|---|---|---|
| Active params | 3B (MoE) | ~29.6B (dense) |
| Modality | Text-only | Text + vision |
| Context | 1M | 131K |
| Quant story | NVFP4 (Blackwell-native) | 4-bit K-Quant + speculative decoding |
| License | OpenMDW-1.1 | Apache 2.0 |
| Target HW | H100/H200/DGX Spark | 24–32GB consumer GPUs |

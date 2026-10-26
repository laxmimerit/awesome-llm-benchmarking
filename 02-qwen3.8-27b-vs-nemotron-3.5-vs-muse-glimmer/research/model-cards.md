# Model cards (as installed, `ollama show`)

All three are Q4_K_M, so the quantization is held constant across the
comparison. Every run used num_ctx 65536, temperature 0.2, num_predict 32768,
on a single RTX 5090 (32 GB) with Ollama 0.32.12.

| | Qwen 3.8 27B | Nemotron 3.5 Lightning | Muse Glimmer |
|---|---|---|---|
| Ollama tag | `qwen3.8:latest` | `nemotron-3.5-lightning:latest` | `muse-glimmer:latest` |
| Architecture | `qwen35` | `nemotron_h_moe` (MoE) | `muse-glimmer` |
| Parameters | 27.3B | 32.9B | 27.9B |
| Quantization | Q4_K_M | Q4_K_M | Q4_K_M |
| Max context | 262,144 | 1,048,576 | 131,072 |
| Embedding length | 5,120 | 2,688 | 6,656 |
| Disk size | 17 GB | 25 GB | 18 GB |
| Vision | yes (CLIP projector, 460.73M) | no | yes |
| Tools | yes | yes | yes |
| Thinking | yes | yes | not advertised |
| Min Ollama | 0.32.12 | 0.32.9 | 0.32.8 |

Notes:

- Nemotron is the only mixture-of-experts model here and the only one with a
  1M-token advertised context. It is also the largest on disk by 7 GB.
- Qwen 3.8 ships a CLIP vision projector (460.73M params) that counts toward
  its footprint even on a pure-text benchmark like this one.
- Muse Glimmer does not advertise a `thinking` capability, so any reasoning it
  does is emitted inline in the response body rather than in a separate
  `thinking` field. Its thinking-word count is therefore not comparable
  like-for-like with the other two.

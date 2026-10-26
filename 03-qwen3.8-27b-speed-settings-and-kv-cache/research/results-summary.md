# Qwen3.8-27B settings sweep - RTX 5090 32GB

- Model: Qwen3.8-27B Q4_K_M (GGUF, MTP head present)
- GPU: RTX 5090 32GB
- llama.cpp build: b10448
- Repeats: 3 (median reported)
- Idle VRAM baseline: 1926.0 MiB (subtracted from all VRAM figures)

| config | decode tok/s | MTP accept | prefill tok/s | TTFT s | model VRAM MiB | quality % | tok/Wh | load s |
|---|---|---|---|---|---|---|---|---|
| llamacpp-ctx32k-kvf16-faon-specnone-think | 73.64 |  | 3036.39 | 0.32 | 17965.9 | 100.0 | 485.3 | 13.67 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn1-think | 104.78 | 0.8599 | 2827.48 | 0.31 | 18648.1 | 100.0 | 690.6 | 13.64 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn2-think | 125.48 | 0.7664 | 2919.58 | 0.26 | 18798.1 | 100.0 | 768.0 | 13.69 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn3-think | 133.63 | 0.6741 | 2910.41 | 0.26 | 18948.1 | 100.0 | 812.0 | 13.73 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think | 119.46 | 0.592 | 2796.32 | 0.29 | 19098.1 | 100.0 | 729.6 | 13.62 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn5-think | 108.81 | 0.5204 | 2723.81 | 0.3 | 19246.1 | 100.0 | 683.9 | 13.68 |
| llamacpp-ctx32k-kvf16-faon-specngram-mod-think | 76.81 | 0.5352 | 2900.85 | 0.32 | 17968.0 | 100.0 | 506.2 | 13.59 |
| llamacpp-ctx32k-kvf16-faon-specnone-nothink | 73.43 |  | 3202.56 | 0.17 | 17965.9 | 100.0 | 462.0 | 13.78 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn2-nothink | 151.24 | 0.8628 | 2937.22 | 0.17 | 18798.1 | 100.0 | 918.9 | 13.77 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn2-think-xhigh | 128.86 | 0.7664 | 2907.53 | 0.27 | 18798.1 | 100.0 | 791.5 | 13.75 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn2-think-medium | 139.24 | 0.8203 | 3004.15 | 0.2 | 18798.1 | 100.0 | 876.1 | 13.68 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn2-think-low | 147.27 | 0.8388 | 2929.11 | 0.29 | 18798.1 | 100.0 | 919.7 | 13.8 |
| llamacpp-ctx32k-kvq8_0-faon-specnone-think | 71.78 |  | 3028.11 | 0.28 | 17091.9 | 100.0 | 478.7 | 12.65 |
| llamacpp-ctx32k-kvq8_0-faon-specdraft-mtpn2-think | 128.32 | 0.7462 | 2947.48 | 0.32 | 17924.1 | 100.0 | 776.8 | 13.78 |
| llamacpp-ctx32k-kvq4_0-faon-specnone-think | 77.12 |  | 3167.48 | 0.17 | 16579.9 | 100.0 | 478.3 | 13.72 |
| llamacpp-ctx32k-kvq4_0-faon-specdraft-mtpn2-think | 136.72 | 0.798 | 2935.34 | 0.31 | 17412.1 | 100.0 | 845.2 | 13.7 |
| llamacpp-ctx32k-kvf16-faoff-specnone-think | 73.19 |  | 2575.92 | 0.46 | 20232.0 | 100.0 | 450.0 | 13.62 |
| ollama-ctx32k-kvf16-faon-specnone-think | 147.24 |  | 3107.43 | 0.34 | 20525.6 | 100.0 | 962.4 | 7.92 |
| llamacpp-ctx4k-kvf16-faon-specdraft-mtpn2-think | 126.46 | 0.768 | 2313.56 | 0.18 | 16862.0 | 100.0 | 769.8 | 13.63 |
| llamacpp-ctx8k-kvf16-faon-specdraft-mtpn2-think | 126.09 | 0.7653 | 2311.61 | 0.19 | 17142.0 | 100.0 | 770.7 | 13.63 |
| llamacpp-ctx16k-kvf16-faon-specdraft-mtpn2-think | 126.25 | 0.7673 | 2677.61 | 0.34 | 17678.0 | 100.0 | 775.2 | 13.63 |
| llamacpp-ctx64k-kvf16-faon-specdraft-mtpn2-think | 126.39 | 0.7659 | 2709.45 | 0.34 | 21008.2 | 100.0 | 775.8 | 13.6 |
| llamacpp-ctx128k-kvq8_0-faon-specdraft-mtpn2-think | 120.62 | 0.7506 | 2212.1 | 0.35 | 22118.5 | 100.0 | 792.7 | 13.67 |
| llamacpp-ctx256k-kvq8_0-faon-specdraft-mtpn2-think | 115.1 | 0.7471 | 587.06 | 0.21 | 27751.0 | 100.0 | 720.6 | 14.65 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn3-think-bson | 130.89 | 0.6741 | 2862.64 | 0.32 | 18948.1 | 100.0 | 867.0 | 13.79 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn3-think-bsoff | 125.99 | 0.6741 | 2840.68 | 0.3 | 18948.1 | 100.0 | 827.1 | 13.72 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-bson | 120.6 | 0.592 | 2834.72 | 0.33 | 19098.1 | 100.0 | 769.8 | 13.73 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-bsoff | 115.99 | 0.592 | 2853.64 | 0.32 | 19098.1 | 100.0 | 746.2 | 13.72 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn3-think-bsoff-rev | 126.0 | 0.6741 | 2861.53 | 0.32 | 18948.1 | 100.0 | 826.2 | 13.79 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn3-think-bson-rev | 130.62 | 0.6741 | 2843.02 | 0.31 | 18948.1 | 100.0 | 853.1 | 13.66 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-bsoff-rev | 116.11 | 0.592 | 2837.71 | 0.29 | 19098.1 | 100.0 | 741.4 | 13.74 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-bson-rev | 120.0 | 0.592 | 2824.19 | 0.33 | 19098.1 | 100.0 | 761.0 | 13.71 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-bson-b1024u1024 | 120.55 | 0.5909 | 2895.64 | 0.29 | 19386.3 | 100.0 | 762.5 | 13.75 |
| llamacpp-ctx32k-kvq8_0-faon-specdraft-mtpn4-think-bson-b1024u1024 | 119.35 | 0.5621 | 2875.42 | 0.31 | 18470.3 | 100.0 | 722.1 | 13.67 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn5-think-rev | 113.04 | 0.5204 | 2829.47 | 0.3 | 19246.1 | 100.0 | 705.1 | 13.74 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-rev | 120.04 | 0.592 | 2839.0 | 0.31 | 19098.1 | 100.0 | 773.6 | 13.75 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn3-think-rev | 129.69 | 0.6741 | 2830.78 | 0.32 | 18948.1 | 100.0 | 856.9 | 13.69 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn2-think-rev | 125.76 | 0.7664 | 2834.21 | 0.3 | 18798.1 | 100.0 | 799.9 | 13.68 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn1-think-rev | 104.98 | 0.8599 | 2840.51 | 0.34 | 18648.1 | 100.0 | 677.0 | 13.72 |
| llamacpp-ctx32k-kvf16-faon-specnone-think-rev | 73.62 |  | 3022.59 | 0.29 | 17965.9 | 100.0 | 483.4 | 12.62 |
| llamacpp-ctx32k-kvf16-faon-specdraft-mtpn4-think-bson-b1024u1024-tmplchatml | 156.49 | 0.6891 | 2976.93 | 0.21 | 19386.3 | 100.0 | 1026.5 | 13.74 |
| llamacpp-ctx32k-kvq8_0-faon-specdraft-mtpn4-think-bson-b1024u1024-tmplchatml | 153.15 | 0.6725 | 2979.98 | 0.12 | 18470.3 | 100.0 | 997.4 | 13.7 |
| ollama-ctx32k-kvq8_0-faon-specnone-think | 154.25 |  | 3096.69 | 0.29 | 19609.6 | 100.0 | 977.1 | 8.48 |
| llamacpp-ctx32k-kvq4_0-faon-specdraft-mtpn3-think | 136.75 | 0.7171 | 2872.39 | 0.31 | 17562.1 | 100.0 | 881.3 | 13.97 |
| llamacpp-ctx128k-kvq8_0-faon-specdraft-mtpn3-think | 121.79 | 0.6563 | 2273.84 | 0.32 | 22268.5 | 100.0 | 761.4 | 13.73 |

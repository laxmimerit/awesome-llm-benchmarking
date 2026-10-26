# Ornith 1.5 benchmark: measured results

### VRAM accounting (Q4_K_M, f16 KV, flash attention on)

| model            | weights MiB | KV layers | KV KiB/token | KV GiB @262k | compute MiB |
| ---------------- | ----------- | --------- | ------------ | ------------ | ----------- |
| ornith-9b        | 5358        | 8         | 32.0         | 8.0          | 232         |
| ornith-35b-a3b   | 20176       | 10        | 20.0         | 5.0          | 204         |
| qwen3.8-27b      | 15770       | 16        | 64.0         | 16.0         | 270         |
| gemma4-31b       | 18216       | 60        | 80.0         | 20.0         | 331         |
| muse-glimmer-30b | 15968       | 52        | 13.0         | 3.2          | 302         |

### Usable context ceiling (decode stays within 90% of normal rate)

| model            | usable max ctx | of advertised | tok/s normal | cliff at | tok/s past cliff | peak VRAM MiB |
| ---------------- | -------------- | ------------- | ------------ | -------- | ---------------- | ------------- |
| ornith-9b        | 262,144        | 100%          | 191.5        | none     | -                | 14640         |
| ornith-35b-a3b   | 262,144        | 100%          | 269.1        | none     | -                | 26765         |
| qwen3.8-27b      | 245,760        | 94%           | 77.3         | 262,144  | 2.9              | 32122         |
| gemma4-31b       | 147,456        | 56%           | 69.7         | 155,648  | 28.3             | 32078         |
| muse-glimmer-30b | 262,144        | 100%          | 79.5         | none     | -                | 20291         |

### KV cache size at 64k context by cache type

| model            | KV f16   | KV q8_0             | KV q4_0             |
| ---------------- | -------- | ------------------- | ------------------- |
| ornith-9b        | 2048 MiB | 1088 MiB (47% less) | 576 MiB (72% less)  |
| ornith-35b-a3b   | 1280 MiB | 680 MiB (47% less)  | 360 MiB (72% less)  |
| qwen3.8-27b      | 4096 MiB | 2176 MiB (47% less) | 1152 MiB (72% less) |
| gemma4-31b       | 6320 MiB | 3358 MiB (47% less) | 1778 MiB (72% less) |
| muse-glimmer-30b | 930 MiB  | 494 MiB (47% less)  | 261 MiB (72% less)  |

### Decode throughput, tok/s (server timings, 512 tokens, MTP off)

| model            | counting | code  | json  | essay | fiction | mean  | spread |
| ---------------- | -------- | ----- | ----- | ----- | ------- | ----- | ------ |
| ornith-9b        | 193.5    | 192.5 | 192.1 | 190.8 | 191.7   | 192.1 | 1.4%   |
| ornith-35b-a3b   | 271.8    | 271.1 | 272.5 | 272.8 | 272.3   | 272.1 | 0.6%   |
| qwen3.8-27b      | 76.8     | 76.9  | 76.8  | 76.6  | 76.6    | 76.7  | 0.3%   |
| gemma4-31b       | 68.9     | 68.8  | 68.8  | 69.0  | 69.1    | 68.9  | 0.4%   |
| muse-glimmer-30b | 79.4     | 79.1  | 79.3  | 79.3  | 79.3    | 79.3  | 0.4%   |

### Prefill throughput, tok/s, by prompt depth

| model            | 512  | 2k   | 8k    | 32k  |
| ---------------- | ---- | ---- | ----- | ---- |
| ornith-9b        | 6176 | 9660 | 10630 | 9921 |
| ornith-35b-a3b   | 1223 | 6726 | 7493  | 7236 |
| qwen3.8-27b      | 2185 | 2896 | 3474  | 3187 |
| gemma4-31b       | 2170 | 2077 | 2901  | 2824 |
| muse-glimmer-30b | 1588 | 3987 | 4088  | 3986 |

### Multi-token prediction (--spec-type draft-mtp)

| model          | content  | off tok/s | on tok/s | speedup | accept | output identical |
| -------------- | -------- | --------- | -------- | ------- | ------ | ---------------- |
| ornith-35b-a3b | counting | 275.7     | 356.4    | 1.29x   | 0.764  | True             |
| ornith-35b-a3b | code     | 275.2     | 303.1    | 1.10x   | 0.629  | False            |
| ornith-35b-a3b | json     | 275.1     | 318.0    | 1.16x   | 0.628  | False            |
| ornith-35b-a3b | essay    | 274.4     | 211.0    | 0.77x   | 0.314  | False            |
| ornith-35b-a3b | fiction  | 274.5     | 205.0    | 0.75x   | 0.297  | False            |
| qwen3.8-27b    | counting | 77.7      | 182.0    | 2.34x   | 0.842  | False            |
| qwen3.8-27b    | code     | 77.5      | 154.9    | 2.00x   | 0.678  | False            |
| qwen3.8-27b    | json     | 77.4      | 182.6    | 2.36x   | 0.857  | False            |
| qwen3.8-27b    | essay    | 77.5      | 95.2     | 1.23x   | 0.313  | False            |
| qwen3.8-27b    | fiction  | 77.4      | 107.2    | 1.38x   | 0.390  | False            |

### Aggregate throughput under concurrency, tok/s

| model            | np=1  | np=4  | np=8  |
| ---------------- | ----- | ----- | ----- |
| ornith-9b        | 185.7 | 439.8 | 508.9 |
| ornith-35b-a3b   | 246.4 | 500.6 | 668.6 |
| qwen3.8-27b      | 74.5  | 203.6 | 259.7 |
| gemma4-31b       | 67.8  | 192.3 | 162.9 |
| muse-glimmer-30b | 77.6  | 234.0 | 287.1 |

### Accuracy and thinking economics

| model            | tier | correct | median tokens | thinking share | median s/answer | truncated |
| ---------------- | ---- | ------- | ------------- | -------------- | --------------- | --------- |
| ornith-9b        | base | 8/8     | 880           | 66%            | 4.6             | 0         |
| ornith-9b        | hard | 8/8     | 3481          | 83%            | 18.1            | 0         |
| ornith-35b-a3b   | base | 8/8     | 845           | 59%            | 3.2             | 0         |
| ornith-35b-a3b   | hard | 8/8     | 2110          | 70%            | 7.8             | 0         |
| qwen3.8-27b      | base | 8/8     | 466           | 87%            | 6.4             | 0         |
| qwen3.8-27b      | hard | 7/8     | 1420          | 89%            | 18.9            | 0         |
| gemma4-31b       | base | 8/8     | 2322          | 73%            | 34.9            | 0         |
| gemma4-31b       | hard | 8/8     | 3948          | 79%            | 59.7            | 0         |
| muse-glimmer-30b | base | 8/8     | 794           | 93%            | 10.1            | 0         |
| muse-glimmer-30b | hard | 8/8     | 1518          | 95%            | 19.3            | 0         |

### Agentic tool use (A1 excluded: defective prompt, see notes)

| model            | base 5 | hard 4 | total | malformed calls | mean tool calls | mean s/task | mean think chars |
| ---------------- | ------ | ------ | ----- | --------------- | --------------- | ----------- | ---------------- |
| ornith-9b        | 10/10  | 8/8    | 18/18 | 0               | 3.8             | 6.2         | 2408             |
| ornith-35b-a3b   | 10/10  | 8/8    | 18/18 | 0               | 4.4             | 4.9         | 2256             |
| qwen3.8-27b      | 10/10  | 8/8    | 18/18 | 0               | 3.6             | 10.6        | 1333             |
| gemma4-31b       | 10/10  | 8/8    | 18/18 | 0               | 3.3             | 16.6        | 2440             |
| muse-glimmer-30b | 10/10  | 8/8    | 18/18 | 0               | 4.7             | 23.4        | 6192             |

### Long-context needle retrieval (found / attempts)

| model            | 16k | 64k | 128k |
| ---------------- | --- | --- | ---- |
| ornith-9b        | 2/2 | 2/2 | 2/2  |
| ornith-35b-a3b   | 2/2 | 2/2 | 2/2  |
| qwen3.8-27b      | 2/2 | 2/2 | 2/2  |
| gemma4-31b       | 2/2 | 2/2 | 2/2  |
| muse-glimmer-30b | 2/2 | 2/2 | 2/2  |
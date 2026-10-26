# Ornith 1.5 benchmark: measured results

### VRAM accounting (Q4_K_M, f16 KV, flash attention on)

| model          | weights MiB | KV layers | KV KiB/token | KV GiB @262k | compute MiB |
| -------------- | ----------- | --------- | ------------ | ------------ | ----------- |
| ornith-35b-a3b | 20176       | 10        | 20.0         | 5.0          | 204         |
| gemma4-26b     | 16808       | 30        | 20.0         | 5.0          | 238         |
| ornith-9b      | 5358        | 8         | 32.0         | 8.0          | 232         |
| qwen3.8-27b    | 15770       | 16        | 64.0         | 16.0         | 270         |

### Usable context ceiling (decode stays within 90% of normal rate)

| model          | usable max ctx | of advertised | tok/s normal | cliff at | tok/s past cliff | peak VRAM MiB |
| -------------- | -------------- | ------------- | ------------ | -------- | ---------------- | ------------- |
| ornith-35b-a3b | 262,144        | 100%          | 271.2        | none     | -                | 27524         |
| gemma4-26b     | 262,144        | 100%          | 235.6        | none     | -                | 23191         |
| ornith-9b      | 262,144        | 100%          | 193.4        | none     | -                | 15444         |
| qwen3.8-27b    | 245,760        | 94%           | 78.4         | 262,144  | 7.3              | 32132         |

### Context ceiling (Q4_K_M, f16 KV, flash attention on)

| model          | max ctx on 32 GB | of advertised 262k | smallest failing ctx   | probes | KV KiB/tok |
| -------------- | ---------------- | ------------------ | ---------------------- | ------ | ---------- |
| ornith-35b-a3b | 262,144          | 100%               | n/a, holds full window | 1      | 20.0       |
| gemma4-26b     | 262,144          | 100%               | n/a, holds full window | 1      | 20.0       |
| ornith-9b      | 262,144          | 100%               | n/a, holds full window | 1      | 32.0       |
| qwen3.8-27b    | 262,144          | 100%               | n/a, holds full window | 2      | 64.0       |

### KV cache size at 64k context by cache type

| model          | KV f16   | KV q8_0             | KV q4_0             |
| -------------- | -------- | ------------------- | ------------------- |
| ornith-35b-a3b | 1280 MiB | 680 MiB (47% less)  | 360 MiB (72% less)  |
| gemma4-26b     | 1580 MiB | 839 MiB (47% less)  | 444 MiB (72% less)  |
| ornith-9b      | 2048 MiB | 1088 MiB (47% less) | 576 MiB (72% less)  |
| qwen3.8-27b    | 4096 MiB | 2176 MiB (47% less) | 1152 MiB (72% less) |

### Decode throughput, tok/s (server timings, 512 tokens, MTP off)

| model          | counting | code  | json  | essay | fiction | mean  | spread |
| -------------- | -------- | ----- | ----- | ----- | ------- | ----- | ------ |
| ornith-35b-a3b | 273.5    | 273.3 | 272.9 | 273.6 | 272.9   | 273.2 | 0.3%   |
| gemma4-26b     | 232.8    | 231.9 | 233.1 | 232.7 | 231.4   | 232.4 | 0.7%   |
| ornith-9b      | 194.7    | 193.0 | 192.2 | 191.7 | 192.1   | 192.7 | 1.5%   |
| qwen3.8-27b    | 77.3     | 77.2  | 77.1  | 77.1  | 77.1    | 77.2  | 0.3%   |

### Prefill throughput, tok/s, by prompt depth

| model          | 512  | 2k   | 8k    | 32k  |
| -------------- | ---- | ---- | ----- | ---- |
| ornith-35b-a3b | 2305 | 6809 | 7547  | 7217 |
| gemma4-26b     | 1114 | 7168 | 9719  | 9533 |
| ornith-9b      | 6362 | 9874 | 10699 | 9952 |
| qwen3.8-27b    | 2270 | 3289 | 3480  | 3184 |

### Multi-token prediction (--spec-type draft-mtp)

| model          | content  | off tok/s | on tok/s | speedup | accept | output identical |
| -------------- | -------- | --------- | -------- | ------- | ------ | ---------------- |
| ornith-9b      | counting | 209.3     | 350.3    | 1.67x   | 0.807  | False            |
| ornith-9b      | code     | 208.6     | 281.5    | 1.35x   | 0.580  | False            |
| ornith-9b      | json     | 208.9     | 232.6    | 1.11x   | 0.430  | False            |
| ornith-9b      | essay    | 208.8     | 176.1    | 0.84x   | 0.261  | False            |
| ornith-9b      | fiction  | 209.5     | 214.9    | 1.03x   | 0.378  | False            |
| ornith-35b-a3b | counting | 273.7     | 351.3    | 1.28x   | 0.764  | True             |
| ornith-35b-a3b | code     | 273.3     | 303.3    | 1.11x   | 0.629  | False            |
| ornith-35b-a3b | json     | 273.0     | 301.4    | 1.10x   | 0.628  | False            |
| ornith-35b-a3b | essay    | 271.1     | 201.1    | 0.74x   | 0.314  | False            |
| ornith-35b-a3b | fiction  | 272.5     | 196.1    | 0.72x   | 0.297  | False            |
| qwen3.8-27b    | counting | 77.3      | 180.6    | 2.33x   | 0.842  | False            |
| qwen3.8-27b    | code     | 77.4      | 154.0    | 1.99x   | 0.678  | False            |
| qwen3.8-27b    | json     | 77.5      | 181.4    | 2.34x   | 0.857  | False            |
| qwen3.8-27b    | essay    | 77.3      | 94.7     | 1.22x   | 0.313  | False            |
| qwen3.8-27b    | fiction  | 77.3      | 107.1    | 1.39x   | 0.390  | False            |

### Aggregate throughput under concurrency, tok/s

| model          | np=1  | np=4  | np=8  |
| -------------- | ----- | ----- | ----- |
| ornith-35b-a3b | 248.9 | 528.7 | 673.0 |
| gemma4-26b     | 222.8 | 546.5 | 702.6 |
| ornith-9b      | 187.4 | 420.4 | 508.2 |
| qwen3.8-27b    | 76.5  | 211.5 | 264.5 |

### Accuracy and thinking economics

| model          | tier | correct | median tokens | thinking share | median s/answer | truncated |
| -------------- | ---- | ------- | ------------- | -------------- | --------------- | --------- |
| ornith-35b-a3b | base | 8/8     | 845           | 59%            | 3.3             | 0         |
| ornith-35b-a3b | hard | 8/8     | 2110          | 70%            | 8.0             | 0         |
| gemma4-26b     | base | 8/8     | 4012          | 83%            | 18.2            | 0         |
| gemma4-26b     | hard | 8/8     | 9759          | 89%            | 43.6            | 0         |
| ornith-9b      | base | 8/8     | 880           | 66%            | 4.8             | 0         |
| ornith-9b      | hard | 8/8     | 3481          | 83%            | 18.8            | 0         |
| qwen3.8-27b    | base | 8/8     | 466           | 87%            | 6.4             | 0         |
| qwen3.8-27b    | hard | 7/8     | 1420          | 89%            | 19.2            | 0         |

### Agentic tool use (A1 excluded: defective prompt, see notes)

| model          | base 5 | hard 4 | total | malformed calls | mean tool calls | mean s/task | mean think chars |
| -------------- | ------ | ------ | ----- | --------------- | --------------- | ----------- | ---------------- |
| ornith-35b-a3b | 10/10  | 8/8    | 18/18 | 0               | 4.5             | 5.1         | 2230             |
| gemma4-26b     | 9/10   | 8/8    | 17/18 | 0               | 3.3             | 10.2        | 5573             |
| ornith-9b      | 10/10  | 8/8    | 18/18 | 0               | 3.6             | 5.9         | 2252             |
| qwen3.8-27b    | 10/10  | 8/8    | 18/18 | 0               | 3.6             | 10.6        | 1281             |

### Long-context needle retrieval (found / attempts)

| model          | 16k | 64k | 128k |
| -------------- | --- | --- | ---- |
| ornith-35b-a3b | 2/2 | 2/2 | 2/2  |
| gemma4-26b     | 2/2 | 2/2 | 2/2  |
| ornith-9b      | 2/2 | 2/2 | 2/2  |
| qwen3.8-27b    | 1/2 | 1/2 | 1/2  |
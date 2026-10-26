# LLM Benchmarking

> **Section 16: Latest LLM Benchmarking and Architecture Teardown 2026 (Lectures 184–197)**

Benchmark harnesses, raw measurements and architecture teardowns behind the
local LLM lectures on the KGP Talkie course and the articles on
[kgptalkie.com](https://kgptalkie.com/tutorials/llm-benchmarking/).

People keep asking how the numbers in those lectures were produced. This
repository is the answer. Every project here is one lecture. Each one carries
the script that ran the sweep, the raw `results` it wrote, the logs from the
run, and the chart code that turned those results into the figures you saw.

Nothing is summarised from memory. If a number appears in a lecture, the file
that produced it is in the matching folder.

## The rig

Unless a project says otherwise, every measurement was taken on one machine:

| Part | Spec |
| --- | --- |
| GPU | NVIDIA RTX 5090, 32 GB VRAM |
| CPU | Intel Core Ultra 9 285K, 24 cores, 24 threads, 8 P plus 16 E, no SMT |
| RAM | 95.3 GB |
| OS | Windows 11 Pro |
| Runtimes | `llama.cpp` (`llama-server`, `llama-bench`) and Ollama |

Two projects are exceptions. Project 11 ran on a MacBook Pro M5 Max, and
project 12 is a desktop agent walkthrough rather than a sweep.

## The projects

| # | Lecture | Project | What it answers | Article |
| --- | --- | --- | --- | --- |
| 01 | 184 | [Nemotron 3.5 vs Muse Glimmer 30B](01-nemotron-3.5-vs-muse-glimmer-30b) | Two new 30B models on one card, on everyday tasks | [read](https://kgptalkie.com/tutorials/llm-benchmarking/nemotron-3-5-lightning-vs-muse-glimmer-30b) |
| 02 | 185 | [Qwen 3.8 27B vs Nemotron 3.5 vs Muse Glimmer](02-qwen3.8-27b-vs-nemotron-3.5-vs-muse-glimmer) | Which 27B class model is worth keeping loaded | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-vs-nemotron-3-5-vs-muse-glimmer) |
| 03 | 186 | [Qwen 3.8 27B speed settings and KV cache](03-qwen3.8-27b-speed-settings-and-kv-cache) | 45 llama.cpp configurations, and which flags actually pay | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-llama-cpp-speed-settings) |
| 04 | 187 | [Qwen 3.8 vs 3.6 vs 3.5 teardown](04-qwen3.8-vs-3.6-vs-3.5-architecture-teardown) | Whether three generations changed the architecture or the training | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-vs-qwen-3-6-27b-vs-qwen-3-5-27b) |
| 05 | 188 | [Qwen 3.8 27B uncensored vs base](05-qwen3.8-27b-uncensored-vs-base) | What an abliterated build physically changes in the weights | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-uncensored-vs-base) |
| 06 | 189 | [Qwen 3.8 Flash Next teardown](06-qwen3.8-flash-next-architecture-teardown) | The four changes from Qwen 3.8 27B to the Qwen 4 preview | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-flash-next-vs-qwen-3-8-27b) |
| 07 | 190 | [Qwen 3.8 Flash Next on CPU only](07-qwen3.8-flash-next-on-cpu-only) | Running a 180B model on one desktop, and what it costs | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-flash-next-on-cpu-only) |
| 08 | 191 | [Ornith 1.5 9B and 35B](08-ornith-1.5-9b-and-35b-vs-qwen3.8-and-gemma4) | A dense 9B and an MoE 35B against the rig anchors | [read](https://kgptalkie.com/tutorials/llm-benchmarking/ornith-1-5-9b-vs-35b-a3b-benchmark) |
| 09 | 192 | [Granite 4.2 vs Qwen 3.8 vs Gemma 4](09-granite-4.2-vs-qwen3.8-27b-and-gemma4) | Two size matched brackets, six models, one build | [read](https://kgptalkie.com/tutorials/llm-benchmarking/granite-4-2-vs-gemma-4-vs-qwen-3-8-benchmark) |
| 10 | 193 | [DeepSeek V4 Flash vs Qwen Flash Next at 1 bit](10-deepseek-v4-flash-vs-qwen3.8-flash-next-on-cpu) | Two huge MoE models at their smallest quantization, on hard problems | [read](https://kgptalkie.com/tutorials/llm-benchmarking/deepseek-v4-flash-1bit-vs-qwen-flash-next) |
| 11 | 194 | [Qwen 3.8 27B on a MacBook Pro M5 Max](11-qwen3.8-27b-on-macbook-pro-m5-max) | The same sweep on Apple Silicon, and what carries across | [read](https://kgptalkie.com/tutorials/llm-benchmarking/qwen-3-8-27b-on-macbook-m5-max) |
| 12 | 195 | [Qwen 3.8 27B with the DeepSeek Harness agent](12-qwen3.8-27b-with-deepseek-harness-agent) | Driving a local agent with Ollama and no API key | |
| 13 | 196 | [DeepSeek V4.1 Flash, DSA and CSA2](13-deepseek-v4.1-flash-dsa-and-csa2-sparse-attention) | How sparse attention picks its 512 entries, from the config | [read](https://kgptalkie.com/tutorials/llm-benchmarking/deepseek-sparse-attention-explained) |
| 14 | 197 | [GLM 5.3 and GLM 5.3 Flash teardown](14-glm-5.3-and-glm-5.3-flash-teardown) | A 744B model and a 320B hybrid, read from their weight maps | [read](https://kgptalkie.com/tutorials/llm-benchmarking/glm-5-3-vs-glm-5-3-flash-architecture-teardown) |

The numbers are the lecture order. Lecture numbers can move when a course is
reorganised, so the folder prefix is what stays stable. Each project README
names its lecture in the first line.

## How a project is laid out

Projects differ, because a speed sweep and an architecture teardown are not the
same kind of work. Most use some of these:

| Path | What is in it |
| --- | --- |
| `research/` | The harness: the sweep driver, the problem sets, the graders, the server control |
| `results/` | Raw measurements as `.jsonl` or `.json`. These are the source of truth |
| `logs/` | Server, download, convert and quantize logs from the actual runs |
| `models/` | Config and weight map JSON only. Never weights |
| `figures/` | The charts used in the lecture, and the `make_figures.py` that draws them from `results/` |
| `PROTOCOL.md` | Rules fixed before any data was collected, where a project has one |

## Two kinds of project

**Measured sweeps** load models and time them. They share a discipline that is
worth copying: one server per configuration, the server killed between every
configuration, the GPU polled until its memory comes back, a unique request id
per prompt so the runtime cannot answer from its own cache, and the idle
baseline re measured before each arm.

**Teardowns** download no weights at all. They fetch a model's `config.json`
and its `model.safetensors.index.json`, sometimes only the headers of the
shards, and count what is actually in the files. A weight map is a list of
every tensor by name and shape, so a block census from it is a count, not a
reading of the model card.

## What is not in this repository

- **Model weights.** No `.gguf`, no `.safetensors`. Every project README names
  the exact repository and quantization it used, so you can pull your own.
- **Compiled runtimes.** No `llama.cpp` build trees, CUDA DLLs or vendored
  checkouts. Build or download the build the project names.
- **Oversized generated dumps.** A few teardowns wrote tensor listings of tens
  of megabytes. The `fetch_headers.py` that writes them is here, so one command
  regenerates them.
- **Third party text.** Papers, vendor blog posts and video transcripts used as
  reading material are linked, not copied. The same goes for evaluation
  corpora: project 05 reads the WikiText-2 test split, and names it rather than
  shipping it.
- **Harmful probe prompts and the answers to them.** Project 05 measured
  refusal behaviour on an abliterated build. Its refusal metrics are published
  in full; the probe prompts and the model answers they produced are not. That
  project's README explains exactly what was held back and what still redraws
  without it.

## Reproducing a run

1. Read that project's README and `PROTOCOL.md` if it has one.
2. Pull the exact model repository and quantization it names.
3. Build or install the runtime build it names. Build numbers matter here. Two
   `llama.cpp` builds a few weeks apart changed our measured speed.
4. Run the driver, usually `research/run_all.py` or a `run_all.sh`.
5. Redraw the charts with `figures/make_figures.py`.

Your absolute numbers will not match ours, and they are not supposed to. A
different card, driver, build or thermal budget moves every throughput figure.
The ratios, the ordering and the shapes of the curves are what should hold.

## A note on units

Sizes here are binary. KB is bytes divided by 1024, MB divided by 1024 squared,
GB divided by 1024 cubed. Where a tool prints MiB we convert and print GB.

## License

Code in this repository is MIT licensed, see [LICENSE](LICENSE). Measurements,
logs and figures are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Model weights,
configs and any vendor material remain under their own licenses.

Many of the models benchmarked here ship under research only or
source available licenses. This repository measures and teaches, it sells
nothing, and a license is never a factor in how a model scored.

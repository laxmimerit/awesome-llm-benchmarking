# Measurement protocol

Fixed before the sweep runs, so nothing here can be tuned after seeing results.

## Runtime

One runtime for every model: `llama-server` from llama.cpp build **b10448**
(CUDA 13.3), the same binary already used for the Qwen 3.8 work in
`../qwen3.8-mtp-benchmark/tools/`.

Ollama is deliberately not used. It is a wrapper that starts this same binary
as a child process with its own undocumented flag defaults (including
speculative decoding), so an "Ollama number" measures Ollama's defaults rather
than the model. Where a model already exists in the local Ollama store, its
blob is a plain GGUF and is fed to `llama-server` directly, which avoids a
duplicate multi-GB download without inheriting Ollama's flags.

## What is held constant

| Knob | Value | Why |
| --- | --- | --- |
| quant | Q4_K_M for all five models | quant class is never a confound |
| offload | `-ngl 999` | every model fully on GPU or the run is void |
| flash attention | `-fa on` | one exception: the explicit on/off sweep |
| KV cache type | f16 | one exception: the explicit KV-quant sweep |
| template | `--jinja`, model's own embedded template | without it `<think>` lands inline and reasoning accounting silently reads zero |
| decode length | `n_predict=512`, `ignore_eos=true` | every model emits exactly the same token count, so no early stop can inflate tok/s |
| prompt caching | off (`cache_prompt=false`) | a cache hit would measure the cache |

Sampling is per model, read from each model card rather than guessed, because a
model's own recommended sampling is a property of that model. It is logged with
every row.

## What is allowed to differ, and the check that makes that legitimate

Multi-token prediction is the one performance knob measured separately per
model, because only three of the five GGUFs carry a `nextn` block.

A performance knob is only a performance knob if it leaves the output alone.
`check_determinism.py` tests that assumption instead of assuming it: two greedy
runs of the same configuration first, to prove the baseline is reproducible at
all, then the MTP run against that baseline.

**Result of that check on Ornith-1.5-9B, recorded before the sweep:** the
baseline is bit-identical run to run, and the MTP run is not. MTP therefore
does not qualify as a free speedup on this build, and its speed numbers are
reported as a separate configuration with its own quality measurement rather
than folded into the headline tok/s.

## Timing

`llama-server`'s own `timings` block is the primary number, with client wall
time recorded alongside as a cross-check. Client-side token counting is not
used, because the OpenAI-compatible path can hide reasoning tokens and inflate
tokens per second.

## Guarding against rig drift

This card is power capped, not thermally capped, under sustained load, and
measured repeatability is about 0.2% within a session against about 6% across
sessions. Two consequences, both binding:

1. Every cross-model speed comparison runs as one uninterrupted block, never
   against a number stored from an earlier session. This is why the Qwen 3.8
   anchor is re-run here rather than quoted from the earlier project.
2. The ladder runs once, forward. A reverse-order control exists as an
   optional phase but is not run by default: run order alone can manufacture
   roughly a 4% effect, and that sits inside the spread the same model shows
   from one machine to another, so it does not change any conclusion drawn
   here. The binding consequence is stated rather than hidden: **any two models
   within about 5% of each other on decode speed are reported as tied, not
   ranked.** If a result lands inside that margin and the ranking matters, the
   reverse pass gets run for that pair specifically.

GPU temperature, power draw, clocks, and the active throttle reason are
recorded on every row.

## Grading

No judge model anywhere. Reasoning problems end in an `ANSWER:` line checked by
a pure function. Agentic tasks are graded by a pure function over the final
answer, against ground truth computed independently from the fixture data, and
every grader is validated against both its expected answer and a plausible
wrong answer before use. The agentic environment never executes model-written
code.

## Models under test

| key | file | source |
| --- | --- | --- |
| ornith-9b | Ornith-1.5-9B-Q4_K_M.gguf | ornith-ai/Ornith-1.5-9B-GGUF |
| ornith-35b-a3b | Ornith-1.5-35B-Q4_K_M.gguf | ornith-ai/Ornith-1.5-35B-A3B-GGUF |
| gemma4-31b | gemma-4-31B-it-Q4_K_M.gguf | unsloth/gemma-4-31B-it-GGUF |
| muse-glimmer-30b | Ollama blob | muse-glimmer:latest |
| qwen3.8-27b | Ollama blob | qwen3.8:latest |

Gemma 4-31B and Muse Glimmer 30B are included because the Ornith 35B card makes
direct head-to-head claims against both. Gemma 4-31B is downloaded rather than
taken from the local Ollama store, whose `gemma4:26b` is a different model to
the one the card names.

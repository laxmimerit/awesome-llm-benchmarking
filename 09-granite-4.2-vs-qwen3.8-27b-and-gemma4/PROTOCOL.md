# Measurement protocol

Fixed before the sweep runs, so nothing here can be tuned after seeing results.

## Question this benchmark answers

Not "which of these six models is smartest". The previous round on this rig
proved that question is unanswerable here: five models scored 39/40 and 18/18
on the old task set, so the eval measured nothing but cost.

The question is **what Granite 4.2 costs to run on one 32 GB card, and whether
its architecture story holds up**, with two three-way brackets as controls.

## Brackets

| Bracket | Models | Weights (Q4_K_M) | Matched on |
| --- | --- | --- | --- |
| A | Granite 4.2 8B, Gemma 4 12B, Ornith 1.5 9B | 5.3 / 7.6 / 5.8 GB | VRAM, **not** parameters (8 / 12 / 9 B) |
| B | Granite 4.2 30B, Gemma 4 31B, Qwen 3.8 27B | 18.0 / 18.3 / 16.8 GB | dense, reasoning, ~17-18 GB |

Bracket A is reported as "what fits in about 8 GB", never as a like-for-like
parameter comparison. Gemma 4 12B also carries vision weights the text tasks
never use. Bracket B is the clean comparison.

## Runtime

One runtime for every model: `llama-server` from llama.cpp build **b10448**,
the same binary used for the Qwen 3.8 and Ornith rounds. Verified before any
download that it loads `general.architecture = granite`; had it not, every
model would have been re-run on a newer build rather than mixing binaries.

Ollama is not used as a runtime. It starts this same binary as a child process
with its own undocumented flag defaults, including speculative decoding, so an
Ollama number measures Ollama's defaults rather than the model. Where a model
exists in the local Ollama store its blob is a plain GGUF and is fed to
`llama-server` directly.

## Held constant

| Knob | Value | Why |
| --- | --- | --- |
| quant | Q4_K_M, all six | quant class is never a confound |
| offload | `-ngl 999` | fully on GPU or the run is void |
| flash attention | `-fa on` | except the explicit on/off sweep |
| KV cache type | f16 | except the explicit KV-quant sweep |
| template | `--jinja`, each model's own embedded template | without it `<think>` lands inline and reasoning accounting reads zero |
| decode length | `n_predict=512`, `ignore_eos=true` for speed runs | identical token counts, so no early stop inflates tok/s |
| prompt caching | off | a cache hit would measure the cache |

Sampling is per model, taken from that model's own card or Modelfile rather
than guessed, and logged with every row.

## Rig drift

The card is power capped, not thermally capped, under sustained load. Cold
runs read 7-8% high and cross-session repeatability is about 6% against 0.2%
within a session. Therefore: every comparison runs back-to-back in one
session, each ladder gets a reverse-order control, and no number from the
earlier Ornith round is reused. Qwen 3.8 and Gemma 4 31B are fully re-run even
though older numbers exist for them.

## Context ceiling

"Largest context that loads" is not measurable on this rig: on Windows a CUDA
allocation past VRAM silently spills to system RAM, so nothing ever fails to
load. The ceiling is defined instead as the largest context where decode stays
within 90% of that model's own small-context rate.

## Quality: what makes this set harder than the last one

Every grader is a pure function. No LLM judge is used anywhere.

* **math** (12 items, 3 tiers) - every answer is produced by a brute-force
  solver committed in `research/problems_hard.py`, and a self-test re-derives
  all 12 before any run.
* **trap** (8 items) - no valid answer exists. The only correct reply is
  IMPOSSIBLE. Scored in the same pass as the math items so a model that hedges
  everything loses the points it would otherwise win.
* **format** (8 items) - graded by a Python predicate over the reply text.
* **code** (9 tasks, 99 hidden asserts) - precise specifications rather than
  interview classics, executed in a separate process. Partial credit recorded
  as asserts passed, which separates "missed one edge case" from "missed the
  task". The reference implementations pass all 99 asserts, checked before any
  model runs.
* **agentic** (8 tasks) - pagination that punishes answering from page one, a
  tool that rejects a malformed key once and explains the fix, and two tasks
  whose data does not exist, where the only correct reply is UNAVAILABLE.
* **multihop** (3 hops at 8k / 32k / 96k fill) - fill lengths measured with
  each model's own tokenizer, since six models with four tokenizers do not
  agree on what "32k tokens" means.

Truncation is recorded separately from wrongness. Any item that hits the token
cap is re-run at double the cap, so "ran out of room" is never reported as
"got it wrong".

## Granite-specific

Granite 4.2 advertises configurable reasoning effort. Whether the dial exists
is settled by rendering the prompt through `/apply-template` and comparing
bytes, not by inferring from output.

## What executes model output

The coding eval runs model-written Python in a short-lived separate
interpreter with a hard timeout. That is stated plainly because it is the only
way to grade code objectively.

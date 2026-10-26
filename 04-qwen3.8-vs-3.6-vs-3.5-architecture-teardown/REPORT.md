# Qwen 27B Teardown: 3.5 vs 3.6 vs 3.8 on Ollama

**RTX 5090 (32 GB), Ollama 0.32.14, Q4_K_M, 397 measured runs, 2026-08-18/19.**

Every accuracy claim below comes from problems with brute-force-verified unique answers.
Every speed claim carries its measurement conditions. Every model comparison was run
forward and reverse (cold-to-hot and hot-to-cold GPU order) with a discarded warmup
generation after every model load. Models were fully unloaded from VRAM before each switch,
verified via /api/ps. Speculative decoding (MTP) was explicitly controlled per run, never
left to per-model defaults.

## 1. Verdict in three sentences

Qwen 3.5, 3.6 and 3.8 at 27B are the same machine: 3.5 and 3.6 are tensor-for-tensor
identical in structure, and 3.8 keeps the identical text backbone. Essentially everything
that improved is post-training: the thinking policy (7x fewer reasoning tokens at equal
accuracy), the MTP draft head (acceptance 0.09 to 0.36 on prose, 0.40 to 0.88 on code,
turning speculative decoding from a net loss into +37% to +160%), and robustness of the
draft-friendly output style. On this suite the practical result is that 3.8 answers
equally correctly roughly 6 to 9 times faster end to end.

## 2. Architecture: proved identical, not assumed

Parsed directly from the GGUF headers of the local Ollama blobs (gguf_dump.py):

| | qwen3.5:27b | qwen3.6:27b | qwen3.8:latest |
|---|---|---|---|
| Architecture tag | qwen35 | qwen35 | qwen35 |
| Parameters | 27,781,427,952 | 27,781,427,952 | ~27.3B + 461M projector |
| Blocks | 64 | 64 | 64 + 1 MTP (`nextn`) |
| Hybrid layout | 3x DeltaNet : 1x full attention (`full_attention_interval 4`) | same | same |
| Hidden / FFN | 5120 / 17408 | same | same |
| Attention heads | 24 Q, 4 KV, head dim 256 | same | same |
| DeltaNet (SSM) | inner 6144, state 128, 48 heads, conv 4 | same | same |
| Vocab | 248,320 | same | same |
| Context | 262,144 | same | same |
| Tensor list | 1307 tensors | **identical names, shapes, dtypes to 3.5** | 866 (vision split out) |
| MTP head | dormant `mtp.*` tensors | dormant `mtp.*` tensors | active `blk.64.nextn.*` |

3.5 vs 3.6: zero differences in tensor names, shapes or dtypes. The weights differ, the
architecture does not. 3.8 repackages: vision moved to a separate 461M mmproj file, and the
multi-token-prediction head ships as a 65th block that Ollama activates by default
(`draft_num_predict 4`). The MTP head is not new in 3.8; it is newly *good* in 3.8 (section 6).

Runtime confirmation: all three decode within 1.5% of each other with MTP off (78.6 to 80.4
tok/s in the content sweep, 70 to 74 in the earlier heat-soaked probe) and show identical
speed-vs-context decay. Identical compute graph, measured twice.

## 3. Accuracy and thinking economics (matched settings)

Matched pass: identical sampling for all three (temp 1.0, top_p 0.95, top_k 20,
presence_penalty 0), think on/off, MTP off, num_ctx 32768, fixed seeds, fwd + rev.
8 problems (4 math, 4 reasoning), all with verified unique answers.

| Model | Think acc | No-think acc | Avg thinking tokens | Avg wall (think) |
|---|---|---|---|---|
| 3.5 | 16/16 | 16/16 | 7,157 | 99.3 s |
| 3.6 | 15/16 | 16/16 | 5,320 | 74.1 s |
| 3.8 | 16/16 | 16/16 | **1,038** | **16.4 s** |

Hard tier (4 harder verified problems, 24,576-token budget, fwd + rev):

| Model | Acc | Avg tokens | Total wall |
|---|---|---|---|
| 3.5 | 8/8 | 13,011 | 23.1 min |
| 3.6 | 7/8 | 13,328 | 23.6 min |
| 3.8 | 8/8 | **1,850** | **3.5 min** |

Two findings that matter more than the scores:

1. **Every failure in the entire 397-run matrix was an overthinker running out of budget,
   never a wrong deduction.** 3.6's two misses (base M3, hard G1) both hit the token cap
   mid-thought. With unlimited budget these models appear to converge; what differs is the
   token bill.
2. **The base problems saturate.** All three are competent 27Bs; base accuracy does not
   separate them. Thinking cost separates them enormously.

## 4. Direction of thinking

Measured on the matched thinking runs:

| Metric | 3.5 | 3.6 | 3.8 |
|---|---|---|---|
| Where the correct answer first appears in the trace | 12% | 16% | 49% |
| Trace remaining after first correct answer (rumination) | 88% | 84% | 51% |
| Self-correction markers per run ("wait", "actually", ...) | 13.8 | 10.0 | 0.6 |
| Verification markers per run | 5.9 | 4.7 | 1.0 |

3.5 finds the right answer early, then spends 7x the tokens it took to find it on loops of
re-verification and second-guessing. 3.6 is the same policy mildly trimmed (25% fewer
tokens). 3.8 is a different policy: near-linear derivation, one verification pass, commit.
The 3.5-to-3.6 step and the 3.6-to-3.8 step are qualitatively different upgrades, which
matches their release framing (3.6 focused on agentic/coding + thinking preservation;
3.8 on a new thinking regime).

## 5. The effort dial: what Ollama actually does (verified in source)

Community reports say qwen3.8 "wildly overthinks" at its default xhigh effort
(22k+ thinking tokens for an SVG drawing, per Simon Willison; an HF discussion titled
"This model cannot stop thinking"). This benchmark measured the opposite: 3.8 never
exceeded ~7k thinking chars. Both are true, and the difference is the serving stack:

- LM Studio / llama-server `--jinja` apply the model's official chat template, whose
  `reasoning_effort` variable defaults to `xhigh` and triggers the trained deep-reasoning
  regime.
- Ollama never uses that template variable. Verified in Ollama v0.32.14 source
  (`model/renderers/qwen35.go`): it prepends a plain-English sentence to the system
  message. `think:"low"` injects "keep your thinking brief"; `think:"medium"` injects
  nothing; `think:"high"` / `"max"` inject a sentence *saying* effort is xhigh;
  `think:true` maps to "medium", i.e. nothing.
- Measured effect of those sentences (8 problems each): low 1,020 tokens, medium 1,115,
  high 600, max 662, all 8/8 correct. The "xhigh" sentence makes thinking *shorter and
  tidier*, not longer. The trained overthinking regime is unreachable through Ollama.

Consequence: every Ollama-served qwen3.8 number (including this report and any public
Ollama benchmark) describes the low-effort regime. That regime is this model's best
feature on consumer hardware, but it is not the regime the model card's headline
benchmarks were run in.

## 6. Speed, MTP, and why "tok/s" claims disagree

Baseline (MTP off): the three models are the same, 78.6 to 80.4 tok/s at 32k ctx in the
content sweep (paired, same session). Sampling penalties cost real speed: presence_penalty
1.5 alone cost ~4% decode.

Speculative decoding is where the versions diverge. Acceptance rate of the draft head,
same engine, same flags, measured per content type (pass H, fwd + rev, paired):

| Content | 3.5 accept | 3.6 accept | 3.8 accept | 3.5 tok/s | 3.6 tok/s | 3.8 tok/s |
|---|---|---|---|---|---|---|
| Counting 1-200 | 0.68 | 0.70 | **1.00** | 158 | 160 | **205** |
| Python + tests | 0.40 | 0.53 | **0.88** | 110 | 133 | **187** |
| Code at temp 0.2 | | | **0.90** | | | **189** |
| JSON records | | | 0.81 | | | 176 |
| Essay prose | 0.09 | 0.12 | 0.36 | **59.5** | **63.2** | 104 |
| Surreal fiction | | | 0.27 | | | 88 |
| (baseline, MTP off) | | | | 80.3 | 79.9 | 78.7 |

Readings:

- **3.8's draft head is better at every point on the curve**, 4x the acceptance on prose.
  This is a post-training gain: the head existed in 3.5/3.6 but predicts its own model's
  output poorly.
- **MTP on 3.5/3.6 is a net loss on prose** (59 to 63 tok/s vs 80 baseline) and a win on
  predictable output (110 to 160). Enabling it is content-dependent for them; for 3.8 it
  never lost in this sweep.
- **Speed claims without content + sampling + MTP state are meaningless.** The historical
  figures for qwen3.8 on this exact machine reconcile as points on this one curve:
  171 tok/s (Aug 14 report: MTP n=4, temp 0.2, code-heavy suite; the code_t02 cell here
  reproduces 189), 125.5 (Aug 16: llama.cpp MTP n=2 f16 ladder), 96.5 to 104 (essay
  prose, acceptance ~0.36), 70 to 79 (MTP off, varying penalties and heat soak).
  No contradiction, one mechanism: tok/s = base speed x MTP factor(acceptance(content,
  temperature, effort, penalties)).

Order controls: fwd vs rev decode differed by under 1% across the accuracy matrix
(warmup after every load; GPU temp before runs 79 to 88 C in both directions). Cross-session
drift of ~6 to 11% between probe sets hours apart is real and documented in the numbers
above; paired same-session cells are the ones to trust, and every comparison table here
is paired.

## 7. Long context: no cliff to 80k, identical decay

Corrected context ladder (relevant reference material prepended; actual prompt tokens
measured; fwd + rev; MTP off; matched thinking):

| Fill (actual tokens) | Accuracy (all 3 models) | Prefill tok/s | Decode tok/s |
|---|---|---|---|
| ~16k | 12/12 | ~3,340 | ~70 |
| ~48k | 12/12 | ~2,910 | ~64 |
| ~80k | 12/12 | ~2,320 | ~59 |

- No accuracy degradation for any model to 80k of context on these tasks.
- Prefill and decode decay curves are identical across the three models (same architecture,
  measured dynamically).
- 3.8 stays terse with a full context window (1.7k to 3.9k thinking chars); 3.5 keeps
  thinking 15k chars regardless of how much context it read.
- 128k num_ctx fits fully on the 32 GB card (30.4 GB used) but leaves little margin with
  desktop apps loaded; 64k is comfortable.

Methodological note: the first ladder attempt sized filler by a 4.0 chars/token estimate;
actual ratios are 3.36 (math filler) and 4.75 (reasoning filler), which pushed some cells
into the context boundary where the server truncates or rejects. That data was archived
(results_D_v1.jsonl) and the ladder rerun with measured ratios. Prompt token counts in the
table are actuals from the server, not estimates.

## 8. Qwen's recommended settings: trust but verify

Per-model official card settings, thinking mode, fwd + rev:

| Model | Matched acc | Recommended acc | What changed |
|---|---|---|---|
| 3.5 | 16/16 | 14/16 | presence_penalty 1.5 (its card's value) |
| 3.6 | 15/16 | 15/16 | identical settings |
| 3.8 | 16/16 | 16/16 | think:"max" (fewer tokens, same acc) |

All three recommended-settings failures were the same event: presence_penalty 1.5 on the
modular-arithmetic problem caused runaway thinking to the 16k cap. Arithmetic verification
is naturally repetitive; a strong presence penalty punishes exactly that repetition and
destabilizes convergence. On Ollama, presence_penalty 0 outperformed 3.5's own card
recommendation. (3.5's Modelfile also ships presence_penalty 1.5 as its Ollama default,
so stock `ollama run qwen3.5` inherits this failure mode; 3.8's Modelfile sets 0.)

## 9. Architecture vs post-training: the attribution

| Gain observed | Source | Evidence |
|---|---|---|
| 7x cheaper thinking at equal accuracy | Post-training | identical graph, different policy; direction-of-thinking metrics |
| MTP acceptance 0.09/0.12 to 0.36+ (prose) | Post-training of the draft head | same head structure present in all three |
| +37% to +160% real decode speedup | Packaging + head training | Ollama activates the (now useful) head by default |
| Long-context robustness to 80k | Already present in 3.5 | all three 12/12 |
| Raw decode/prefill speed | Unchanged | identical within 1.5%, identical decay |
| Multimodal packaging | Repackaging | vision split to mmproj |

The one caveat: through Ollama the official xhigh deep-thinking regime is unreachable, so
this comparison characterizes the low-effort regime. Community and prior local llama.cpp
data confirm 3.8 can also think 20k+ tokens under its official template; whether *that*
regime buys accuracy on harder problems than this suite is an open question for the vLLM
follow-up (which will apply the official template).

## 10. Practical recommendations (Ollama, this hardware)

- **Default choice: qwen3.8.** Same accuracy as its predecessors here, 6 to 9x faster in
  practice, MTP on by default, and 176 to 205 tok/s on structured output.
- Leave 3.8's MTP on (never lost in the sweep). For 3.5/3.6, MTP helps only on
  code/structured output; on prose it costs ~25%.
- Use presence_penalty 0 in thinking mode for all three, including 3.5, despite its card.
- The `think` levels through Ollama are weak text nudges: "high"/"max" shortens thinking
  slightly, "low" barely differs from default. Do not expect the model card's
  reasoning-effort semantics.
- 64k context is free in quality and cheap in speed (-8%); 128k fits but is tight on VRAM
  with desktop apps open.

## 11. Method and limitations

- 397 runs total. Problems: 8 base + 4 hard, answers brute-force verified unique
  (verify_problems.py, verify_hard.py); graders are string/regex extractors on a mandated
  ANSWER line, never LLM judges. Filler text verified free of answer leakage.
- Controls: forward + reverse model order per pass; discarded warmup after every load;
  explicit VRAM unload between models (polled to empty); MTP pinned per run; fixed seeds;
  GPU temp/power/throttle logged before every run; server-reported token timings.
- Accuracy cells are 2 samples (fwd + rev) at temp 1.0: fine for the large effects here
  (7x token gaps), not for 1-problem differences. 3.5 vs 3.6 accuracy is a tie within
  noise; the token economics are 16-sample-consistent.
- One Ollama serve session (0.32.14, debug logging on) on one machine, Q4_K_M only.
  Cross-session drift on this rig is ~6%; all comparison tables are same-session paired.
- Ollama runner IPC produced sporadic instant HTTP 500s on the first large-prompt request
  after a model swap (server log: "chat prompt error ... wsarecv: connection forcibly
  closed"). Retried automatically; one cell needed a second invocation. No run was graded
  from a failed request.
- Full raw data: results.jsonl (+ archived results_D_v1.jsonl), harness bench.py,
  problems.py, analysis analyze.py, GGUF metadata gguf_meta.json.

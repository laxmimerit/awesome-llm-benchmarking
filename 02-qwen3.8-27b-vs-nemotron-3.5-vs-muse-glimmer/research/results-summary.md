# Qwen 3.8 27B vs Nemotron 3.5 Lightning vs Muse Glimmer

**Hard-problem benchmark v2 — RTX 5090, Ollama 0.32.12, 64K context, Q4_K_M,
2026-08-14.**

This is also the table-view twin of the charts: every value plotted in
`blog/*.png` appears here as a number.

Exclusive loading was enforced throughout. Before each model loaded, every
model reported by `/api/tags` was unloaded and `/api/ps` was polled until it
reported nothing resident. Verified live mid-run: `ollama ps` showed only the
model under test, at 100% GPU.

## Scoreboard (16 hard tasks)

| Model | Score | Wrong | Out of budget | VRAM @64K | Avg gen tok/s | Total wall | Gen tokens | Thinking words |
|---|---|---|---|---|---|---|---|---|
| **Muse Glimmer** (27.9B) | **15/16** | 0 | 1 | 16.9 GB | 81 | 16.2 min | 75,197 | 32,392 |
| Nemotron 3.5 Lightning (32.9B MoE) | 14/16 | 0 | 2 | 25.4 GB | **216** | 8.3 min | 102,736 | 43,487 |
| Qwen 3.8 27B (27.3B) | 14/16 | 2 | 0 | 17.5 GB | 171 | **6.3 min** | 59,524 | 23,812 |

Glimmer wins on accuracy. Qwen 3.8 wins on time and is the only model that
never ran out of generation budget. Nemotron wins on raw token throughput but
pays for it in VRAM (25.4 GB of the 5090's 32 GB).

## Failures — and they are not the same kind of failure

| Model | Task | Failure mode | Detail |
|---|---|---|---|
| Muse Glimmer | 4×10 domino tilings | out of budget | hit the 32,768-token ceiling |
| Nemotron | Repunit digit sum | out of budget | hit the ceiling |
| Nemotron | Expression evaluator | out of budget | hit the ceiling |
| Qwen 3.8 | Repunit digit sum | **wrong answer** | said 1229, truth is 1230 |
| Qwen 3.8 | 4×10 domino tilings | **wrong answer** | said 17305, truth is 18061 |

**Three of the five failures were generation-cap exhaustion, not reasoning
errors.** The models did not get these wrong; they were still working when the
budget ran out. This is the single most important methodological point in the
run, and it is why `done_reason` is recorded per task.

The two Qwen failures are the opposite problem — it finished comfortably inside
budget and was confidently wrong. On the tilings task it answered 17305 and
then asserted "this matches the known sequence for 4×n domino tilings (OEIS
A005178)". It does not. A005178 runs …2245, 6336, **18061**, and Qwen's own
working table shows a(4)=35 where the true sequence has 36, so its recurrence
drifted from n=4 onward. The confident citation of a real OEIS sequence for a
wrong number is a more dangerous failure than running out of room.

On the repunit task Qwen was off by exactly 1 after 27,179 tokens of work.

## Supplementary diagnostic: does a bigger budget rescue the capped tasks?

Not part of the headline comparison — the scoreboard above stands at an equal
32,768-token budget for all three models. This re-ran **only** the three capped
model/task pairs with `num_predict` raised to 60,000, to separate "cannot solve
it" from "just needed more room". Same 64K context, same VRAM.

| Model | Task | At 60,000 tokens | Verdict |
|---|---|---|---|
| Nemotron | Repunit digit sum | still hit the ceiling (60,000 tok) | runaway reasoning |
| Nemotron | Expression evaluator | terminated at 29,660 tok, still failed | genuine capability failure |
| Muse Glimmer | 4×10 domino tilings | still hit the ceiling (60,000 tok) | runaway reasoning |

**Nearly doubling the budget rescued zero of the three.** Two of them consumed
60,000 tokens without terminating, which is runaway reasoning rather than a
ceiling set slightly too low. So the original 32,768 cap was not the thing
holding these models back, and the headline scores are not an artifact of a
stingy budget.

One honest caveat: Nemotron's expression evaluator hit the cap at 32,768 in the
main run but terminated at 29,660 tokens in this one. At temperature 0.2 the
two runs took different paths — so that task's *main-run* classification as
"out of budget" and its *diagnostic* classification as "wrong" are both real
observations of an unstable task, not a contradiction.

## What every model got right

All three solved 13 of the 16 tasks, including every Code task except
Nemotron's expression evaluator, and every Reasoning task:

- 7^(7^7) mod 1000 → 343 (a power tower, and the clean-looking answer is
  genuinely correct)
- Expected rolls to three consecutive sixes → 258
- MISSISSIPPI arrangements with no two S adjacent → 7,350
- Expected draws to first blue ball → 9/4
- Count inversions in O(n log n), gated at 300,000 elements under 6 s
- O(1) LRU cache, gated at 200,000 mixed operations under 6 s
- Closure late-binding fix
- Version-string comparison
- **Predicting exact interpreter output**, including `list(zip(g, g))` on a
  single generator evaluating to `[(1, 2)]`
- 5×5 logic grid
- PYTHON acrostic with exactly 7 words per sentence
- JSON with computed subtotal, 10% discount and 18% GST on the discounted
  amount
- 64K three-hop needle with deliberately misleading decoy notes

The `zip(g, g)` prediction is worth calling out. In the previous 12-task run
only Glimmer solved the (easier) output-prediction task; here all three solved
a harder one.

## Category breakdown

| Category | Tasks | Nemotron | Glimmer | Qwen 3.8 |
|---|---|---|---|---|
| Math | 6 | 5 | 5 | 4 |
| Code | 6 | 5 | 6 | 6 |
| Reasoning | 4 | 4 | 4 | 4 |

The three models are indistinguishable on Code and Reasoning at this
difficulty. **Math is the only category that separated them**, which means a
harder Math set is where the next round of difficulty should go.

## Where Qwen 3.8 27B comes out ahead

**On accuracy, one task.** Qwen 3.8 solved exactly one task that a rival
missed — the expression evaluator, which Nemotron failed. It never solved
anything Muse Glimmer missed. Against Glimmer its accuracy record is 0 wins,
1 loss (repunit digit sum); against Nemotron it is 1 win, 1 loss (4×10
tilings), which is how both land on 14/16.

**On latency, decisively.** Across the 13 tasks all three models solved, Qwen
3.8 finished first on 8, Nemotron on 5, Glimmer on 0.

| Task | Qwen 3.8 | Best rival | Advantage |
|---|---|---|---|
| PYTHON acrostic, 7 words | 5.6 s | 11.4 s (Nemotron) | **+5.8 s** |
| Predict exact output | 4.4 s | 10.0 s (Glimmer) | **+5.6 s** |
| O(1) LRU cache | 5.9 s | 11.2 s (Nemotron) | **+5.3 s** |
| JSON with computed totals | 4.0 s | 9.2 s (Nemotron) | **+5.2 s** |
| 5×5 logic grid | 7.4 s | 11.1 s (Nemotron) | **+3.7 s** |
| Closure late-binding fix | 1.8 s | 3.2 s (Nemotron) | **+1.4 s** |
| MISSISSIPPI, no two S | 4.6 s | 5.4 s (Nemotron) | **+0.8 s** |
| Version comparison | 3.7 s | 4.0 s (Nemotron) | **+0.3 s** |
| 7^(7^7) mod 1000 | 12.5 s | 12.2 s (Nemotron) | −0.3 s |
| E[rolls to three 6s] | 8.9 s | 6.4 s (Nemotron) | −2.5 s |
| Count inversions | 20.3 s | 11.6 s (Nemotron) | −8.7 s |
| 64K three-hop needle | 28.9 s | 10.6 s (Nemotron) | −18.3 s |
| E[draws to first blue] | 32.1 s | 10.6 s (Nemotron) | −21.5 s |

The shape of that table matters more than the count. Qwen's wins are the
**short, well-specified tasks** — formatting, structured output, a bug fix, a
cache implementation — where it answers in 2–7 seconds. Its losses are the
**long-reasoning tasks** — a 64K retrieval, an expected-value derivation, an
O(n log n) complexity gate — where it spends far longer than Nemotron.

By category, 4 of the 8 wins are Code and 3 are Reasoning; only 1 is Math.
Of the 5 losses, 3 are Math. That matches the category scoreboard.

**Aggregate on the shared tasks actually favours Nemotron**, and this is worth
stating plainly because the per-task win count alone would mislead:

| | Total over the 13 shared tasks | Median task |
|---|---|---|
| Qwen 3.8 | 140.1 s | **5.9 s** |
| Nemotron 3.5 | **130.2 s** | 10.6 s |
| Muse Glimmer | 205.3 s | 14.5 s |

Qwen's 8 wins are worth 28.1 s combined; its 5 losses cost 51.3 s. It wins
many small races and loses a few big ones, so Nemotron takes the shared-task
total by ~10 s while Qwen takes the median by nearly 2×.

Qwen's whole-suite lead (6.3 min vs 8.3 min) therefore does **not** come from
being faster task-for-task. It comes from never stalling: Nemotron spent
310.9 s across its two capped tasks and returned nothing usable from either,
which is more than the entire margin between them.

Three other places Qwen 3.8 led outright:

- **Zero generation-cap hits**, the only model with none.
- **Fewest generated tokens**: 59,524, against 75,197 (Glimmer) and 102,736
  (Nemotron).
- **Least thinking**: 23,812 words, roughly half Nemotron's 43,487.

The honest summary: Qwen 3.8 is the most efficient model here, not the most
accurate one. It is the pick when latency and token cost matter; Glimmer is
the pick when being right matters most.

## Method notes

- **Every ground truth was computed, never hand-derived** (`verify_truth.py`):
  1230, 343, 258, 18061, 7350, 9/4, 777, and a brute-forced check that the 5×5
  logic grid has exactly one solution. The tiling DP reproduces the known
  2×12 = 233 and the OEIS A005178 sequence, so it is validated independently.
- **Every grader was self-tested against known-bad input** (`verify_tasks.py`,
  34/34 passing). It confirms the graders *reject* an O(n²) inversion counter,
  an LRU with no recency tracking, string-comparison version sorting, and
  left-associative `**`. The expected interpreter output is obtained by
  actually executing the snippet, not by assumption.
- **"Do not use eval" is enforced statically**, because an `eval`-based
  calculator passes every execution test. The self-test verifies both that the
  static rule rejects it and that it would otherwise have passed.
- Code answers are graded by execution in a subprocess with timeouts;
  algorithmic complexity is enforced by size-and-time gates rather than by
  reading the code.
- num_ctx 65536, temperature 0.2, num_predict 32768, single run per task.

## Limitations

- **Single run per task at temperature 0.2.** These are not error bars. A
  one-task gap between models is within what a re-run could plausibly move;
  the failure-mode differences are the more robust finding.
- Muse Glimmer does not advertise a `thinking` capability, so its reasoning is
  emitted inline in the response body. Its thinking-word count is therefore
  **not** like-for-like with the other two and should not be compared directly.
- Nemotron carries a vision-free architecture while Qwen 3.8 and Glimmer both
  ship vision projectors, which count toward footprint even on this text-only
  suite.
- The previous 12-task suite is a *different* suite, so scores are not
  comparable across the two posts. Only same-suite numbers here should be
  compared.

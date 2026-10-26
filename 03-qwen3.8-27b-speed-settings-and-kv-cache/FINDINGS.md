# Qwen3.8-27B on an RTX 5090: what the settings actually do

Measured on a single RTX 5090 (32 GB), Qwen3.8-27B Q4_K_M (the Ollama blob,
MTP head intact), llama.cpp build **b10448**, Ollama **0.32.13**.
45 configurations, median of 3 repeats, warmup discarded.

Status: **complete**. Every axis includes a reverse-order control run, because
the card is power-capped and run order alone can manufacture a 4% effect.

---

## 1. The premise checks out; almost every specific claim does not

The viral thread's central claim is real: Qwen3.8 ships a multi-token-prediction
draft head inside the GGUF, and one llama.cpp flag activates it for a large,
quality-free speedup. Verified directly in the model file:

```
qwen35.nextn_predict_layers = 1
blk.64.nextn.eh_proj.weight      [10240, 5120]  Q4_K
blk.64.nextn.enorm.weight        [5120]         F32
blk.64.nextn.hnorm.weight        [5120]         F32
blk.64.nextn.shared_head_norm.weight [5120]     F32
blk indices 0..64   (64 transformer layers + 1 MTP layer)
qwen35.context_length = 262144
```

Everything below is measured against that.

| Claim in the thread | Verdict |
|---|---|
| `--spec-type draft-mtp` gives a big free speedup | **True**, up to 1.81x |
| The draft head has exactly one layer | **True**, `nextn_predict_layers = 1` |
| `--spec-draft-type-k/v` exist | **True** (aliases of `--cache-type-k/v-draft`) |
| Reasoning defaults to `xhigh`; set `medium` | **True**, the one tuning tip that holds |
| "n=2 is the sweet spot" | **Partly**, n=3 wins on an f16 cache (1.81x vs 1.70x), but ties with n=2 on q4_0, where n=2 is the better pick |
| "n=3 is the ceiling" | **Wrong**, it is the peak, not a limit |
| "n=4 does not exist / crashes the head / emits junk tokens" | **False**, runs at 100% quality |
| "n=1 1.75x, n=2 2.37x, n=3 2.85x" | **Not reproduced**, 1.42x / 1.70x / 1.81x |
| Use `q8_0` KV cache | **Beaten by q4_0**, less memory *and* faster |
| `--kv-cache-dtype bfloat16` fixes long-context quality | **Not a llama.cpp flag** (it is vLLM); no quality loss to fix |
| Context is 1M tokens | **262,144** per the model's own metadata |

---

## 2. Speculative decoding: the ladder

32K context, f16 KV cache, reasoning on, steady-state GPU.

| setting | decode tok/s | vs off | accept rate | tokens/step | VRAM GB | tok/Wh | quality |
|---|---|---|---|---|---|---|---|
| MTP off | 73.6 | - | - | - | 17.54 | 485 | 100% |
| `n=1` | 104.8 | 1.42x | 0.860 | 1.88 | 18.21 | 691 | 100% |
| `n=2` | 125.5 | 1.70x | 0.766 | 2.56 | 18.36 | 768 | 100% |
| **`n=3`** | **133.6** | **1.81x** | 0.674 | 3.11 | 18.50 | **812** | 100% |
| `n=4` | 119.5 | 1.62x | 0.592 | 3.56 | 18.65 | 730 | 100% |
| `n=5` | 108.8 | 1.48x | 0.520 | 3.90 | 18.79 | 684 | 100% |
| `ngram-mod` | 76.8 | 1.04x | 0.535 | 26.92 | 17.55 | 506 | 100% |

**The mechanism, in one line:** as draft depth rises, per-token acceptance falls
(0.86 -> 0.52) while accepted tokens per step rises (1.88 -> 3.90). The product of
those two curves peaks at n=3.

**Nothing breaks at any depth.** n=4 and n=5 answer every task correctly.
Speculative decoding is mathematically lossless, a drafted token is only kept if
the full model would have produced it anyway, so depth can cost speed, never
accuracy. The claim that n=4 "starts emitting junk tokens" misdescribes what
speculative decoding is.

**MTP costs memory and prefill.** The draft context adds ~680 MB at n=1, growing
~150 MB per level. Prompt processing drops ~5-8% (3036 -> ~2900 tok/s), consistent
with the device-to-host embedding transfer the upstream repo mentions.

**n-gram drafting is not worth enabling here**, and it is a benchmarking trap.
It scored 1.04x, but only because it *learns from text it has already produced*:

```
decode_structured   rep0=76   rep1=97   rep2=699 tok/s
mean accepted length: 26.92 tokens
```

No real drafter accepts 27-token runs; that is cached-response replay. On cold,
novel prompts it delivers **0.97x**, slightly *slower* than no speculation at
all (71.15 vs 73.56 tok/s on first pass). Its entire apparent benefit comes from
answering prompts it has already seen. Any benchmark that loops identical prompts
will wildly overstate n-gram speculation.

MTP is immune to this: it is stateless per request, and its acceptance rates came
out bit-identical (0.8599, 0.7664) across two fully independent runs. Every MTP
config's cold figure agrees with its median to within 2% (worst case n=5, +1.9%;
n=1 is -0.1%), and every one of those deviations is *positive*, the cold run is
marginally the fastest, which is the power-cap signature, not a warming cache.

### The n=3 peak is not an artefact of run order

The ladder above was measured n=1 -> n=5 in sequence. The 5090 is power-capped
under sustained load (599 W of a 600 W limit, clocks falling from 3090 to 2700
MHz), so each successive config met a hotter, slower GPU, which is exactly the
gradient needed to manufacture a peak at whichever setting ran early. The whole
ladder was therefore re-run in reverse, n=5 -> n=1:

| setting | n=1→n=5 order | n=5→n=1 order | change | accept rate |
|---|---|---|---|---|
| MTP off | 73.6 | 73.6 | -0.0% | - |
| `n=1` | 104.8 | 105.0 | +0.2% | 0.8599 / 0.8599 |
| `n=2` | 125.5 | 125.8 | +0.2% | 0.7664 / 0.7664 |
| **`n=3`** | **133.6** | **129.7** | -2.9% | 0.6741 / 0.6741 |
| `n=4` | 119.5 | 120.0 | +0.5% | 0.5920 / 0.5920 |
| `n=5` | 108.8 | 113.0 | **+3.9%** | 0.5204 / 0.5204 |

**n=3 still peaks**, in both directions. The ordering bias is real but confined
to the tail: n=5 ran last in the forward sweep and was under-measured by 3.9%.
Acceptance rates are bit-identical across the two runs, as they must be, they
are a property of the draft head, not of temperature.

The honest summary is that n=3 is the optimum and n=2 is within ~5% of it, while
the *exact* peak height carries roughly ±3% of run-order uncertainty. Anyone
publishing a single-pass ladder on a power-capped card is measuring their cooling
as much as their settings.

---

## 3. The biggest lever is not in the thread at all

Qwen3.8 reasons by default. Turning that off moves performance more than any
server flag.

| mode | spec | think tokens | **time to answer** | decode tok/s | accept | quality |
|---|---|---|---|---|---|---|
| thinking on | none | 106 | 2.18 s | 73.6 | - | 100% |
| thinking on | MTP n=3 | 106 | 1.30 s | 133.6 | 0.674 | 100% |
| thinking **off** | none | 0 | 0.18 s | 73.4 | - | 100% |
| thinking **off** | MTP n=2 | 0 | **0.21 s** | **151.2** | **0.863** | 100% |

- **Time to first answer improves ~10x** (2.18 s -> 0.21 s). The user waits through
  106 reasoning tokens before seeing any answer.
- **Thinking also degrades MTP.** Acceptance rises 0.766 -> 0.863 with reasoning
  off, because reasoning prose is less predictable than direct answers. The gains
  compound: 73.4 -> 151.2 tok/s, a **2.06x** speedup, better than MTP alone.

### Reasoning effort levels

Qwen3.8's chat template accepts exactly `low`, `medium`, `xhigh` (default) and
**raises an exception on anything else**. llama.cpp's `--reasoning-effort` also
offers `minimal`, `high`, `max`, passing those to this model is an error.

| level | think tokens | time to answer | decode tok/s | accept | quality |
|---|---|---|---|---|---|
| `xhigh` (default) | 106 | 1.44 s | 128.9 | 0.766 | 100% |
| `medium` | 91 | **0.92 s** | 139.2 | 0.820 | 100% |
| `low` | 78 | 1.21 s | 147.3 | 0.839 | 100% |
| off | 0 | 0.21 s | 151.2 | 0.863 | 100% |

Thinking tokens, decode speed and acceptance all move monotonically. `medium` is
a free win, exactly as the thread claims.

**The level list is enforced by the model, not by convention.** Probing a live
server confirms it: `xhigh` returns output byte-identical to the template
default (86 tokens, 110 thinking chars, so `xhigh` genuinely *is* the
default), while `minimal`, `high` and `max` are rejected outright by the
template's own guard:

```
HTTP 500: While executing CallExpression at line 49 ...
          raise_exception('Unexpected reasoning effort ' ~ reasoning_effort)
```

llama.cpp's `--reasoning-effort` accepts all six names, so the failure surfaces
only at request time, per request.

*Caveat, and it is a real one:* `low` shows a higher time-to-answer than
`medium` despite fewer thinking tokens, against the trend in every other
column. The single-prompt probe above disagrees with the table's ordering too
(110 / 193 / 164 thinking chars for `xhigh` / `medium` / `low`). The table's
figures are medians over a workload suite and the probe is one short arithmetic
question, so the suite numbers are the more reliable of the two, but the
disagreement says the effort levels do not shorten reasoning uniformly across
prompts. Treat the *direction* (lower effort -> less thinking, faster decode,
higher acceptance) as established and the per-level numbers as
prompt-dependent.

---

## 4. KV cache: quantise it, and use q4_0

32K context, MTP n=2 where shown.

| KV type | spec | decode tok/s | VRAM GB | saved vs f16 GB | quality | needles d10/d50/d90 |
|---|---|---|---|---|---|---|
| f16 | none | 73.6 | 17.54 | - | 100% | all pass |
| f16 | MTP n=2 | 125.5 | 18.36 | - | 100% | all pass |
| q8_0 | none | 71.8 | 16.69 | 0.85 | 100% | all pass |
| q8_0 | MTP n=2 | 128.3 | 17.50 | 0.85 | 100% | all pass |
| q4_0 | none | 77.1 | 16.19 | 1.35 | 100% | all pass |
| **q4_0** | **MTP n=2** | **136.7** | **17.00** | **1.35** | **100%** | **all pass** |

**Quantising the cache to `q4_0` makes it faster, not slower.** Decoding is
memory-bandwidth bound, so a smaller KV cache means less data read per token.
`q4_0` + `n=2` (136.7 tok/s) beats `f16` + `n=3` (133.6), the best result from
the whole spec sweep, while using 1.35 GB less memory.

The effect is not uniform across precisions. `q8_0` without a draft head
decodes at 71.8 tok/s, *slower* than `f16` at 73.6, and only overtakes it once
MTP is on (128.3 against 125.5). Only `q4_0` is faster at both.

**No quality cost at any precision**, including needle retrieval at three depths.

**`q8_0` does not halve the cache.** Measured, it is ~63% of f16 (43 vs 68 MB per
1K tokens), not 50%.

---

## 4a. Backend sampling: already on, don't turn it off

Reading Ollama's command line surfaced `--spec-draft-backend-sampling`, which
moves draft-token sampling onto the GPU instead of round-tripping to the host.
It looked like a free tuning win. It is not a win, because it is already the
default, `--log-verbosity 4` prints `backend_sampling=1` when the flag is
absent, so Ollama passing it explicitly is a no-op. What the measurement
actually shows is the cost of *disabling* it:

| draft n | backend sampling | decode tok/s | accept rate | mean len |
|---|---|---|---|---|
| `n=3` | on (default) | 130.9 | 0.6741 | 3.114 |
| `n=3` | `--no-spec-draft-backend-sampling` | 126.0 | 0.6741 | 3.114 |
| `n=4` | on (default) | 120.6 | 0.5920 | 3.561 |
| `n=4` | `--no-spec-draft-backend-sampling` | 116.0 | 0.5920 | 3.561 |

Turning it off costs **3.7-4.0%**. Acceptance rate and mean accepted length are
bit-identical with it on and off, which is the expected signature: the flag
changes *where* sampling runs, not what gets sampled.

This pair was also run in both orders, because `on` was measured first in every
original pair and would have inherited the cooler GPU. Reversing changed
nothing, `n=3` off measured 125.99 forward and 126.00 reverse. **Within a
single session the harness reproduces to ~0.2%**; the meaningful drift is
between sessions, where two identical configs differed by 6%. Compare within a
session, never across.

---

## 5. Flash attention: keep it on

| | `-fa on` | `-fa off` |
|---|---|---|
| VRAM | **17.54 GB** | 19.76 GB |
| prefill | **3036 tok/s** | 2576 tok/s |
| TTFT | **0.32 s** | 0.46 s |
| decode | 73.6 | 73.2 |

Saves **2.3 GB** and 20% of prompt-processing time; costs nothing.

---

## 6. Context length: memory is fine, latency is not

MTP n=2, f16 up to 64K then q8_0 (f16 would not fit).

| ctx | VRAM | decode (short prompt) | full prompt | **TTFT (full)** | decode (full) | quality |
|---|---|---|---|---|---|---|
| 4K | 16.5 GB | 126.5 | 3,213 | 1.4 s | 136.3 | 100% |
| 8K | 16.7 | 126.1 | 6,979 | 2.7 s | 121.5 | 100% |
| 16K | 17.3 | 126.3 | 14,500 | 5.2 s | 124.9 | 100% |
| 32K | 18.4 | 126.5 | 29,973 | 11 s | 122.7 | 100% |
| 64K | 20.5 | 126.4 | 61,532 | 27 s | 109.7 | 100% |
| 128K | 21.6 | 120.6 | 124,626 | 78 s | 77.2 | 100% |
| 256K | 27.1 | 115.1 | 252,511 | **757 s** | 13.3 | 100% |

Two different questions that benchmarks routinely conflate:

- **Raising `-c` is nearly free in speed.** Decode holds at ~126 tok/s from 4K to
  64K. It costs VRAM, not throughput.
- **Filling the window is brutally expensive, and superlinear.** Doubling 128K ->
  256K multiplied time-to-first-token by **9.7x**, not 2x.

**256K genuinely works on a 32 GB card**, 27.1 GB resident, perfect needle recall
at all depths, with a q8_0 cache. But you wait **12.5 minutes** for the first
token. The practical interactive ceiling is 64K-128K.

**Long-context servers need warming.** At 256K the first requests after load run
at roughly half speed (53.8 -> 115.1 tok/s by the second pass) while the 5.5 GB
cache is committed.

---

## 6a. Ollama vs llama.cpp: the gap is the prompt, not the engine

Run out of the box against each other, Ollama decodes **2.0x** faster than
`llama-server`. That number is reproducible and completely misleading. Ollama
*is* llama.cpp, it launches `llama-server` as a child process, so the
comparison can only ever be measuring configuration. Adding Ollama's flags to
stock llama.cpp one group at a time shows where the gap comes from:

Throughput below is `completion_tokens / wall_s`, which is measured identically
on both runtimes. The per-runtime `decode_tps` figures elsewhere in this
document are **not** comparable across runtimes, Ollama's come from its own
`eval_duration`, llama.cpp's from the client stream, and a like-for-like check
on a single response put the two ~3.8% apart.

| config | tok/s | vs stock | accept rate |
|---|---|---|---|
| llama.cpp, stock defaults | 70.7 | 1.00x | - |
| + Ollama's MTP flags (`draft-mtp`, `n=4`, `-b/-ub 1024`) | 115.5 | 1.63x | 0.591 |
| + Ollama's `chatml` template | **147.8** | **2.09x** | 0.689 |
| Ollama, default | 133.5 | 1.89x | 0.815 |

**Two thirds of the gap is simply that Ollama enables MTP and llama.cpp does
not.** The rest is the prompt. Fully matched, llama.cpp is not slower than
Ollama at all, it is ~11% *faster*.

The prompt difference is measurable rather than inferred. Ollama's model
template is a bare passthrough (`ollama show --template` returns
`{{ .Prompt }}`) and it hands chat formatting to its child via
`--chat-template chatml`, so Qwen3.8's own Jinja template, the thing that
carries the `reasoning_effort` directive, is never applied. Reading the draft
acceptance its child logs under this benchmark's own workloads:

```
Ollama    draft acceptance = 0.81458 (391 accepted / 480 generated), mean len = 4.26
llama.cpp draft acceptance = 0.5909                                 , mean len = 3.56
```

Same model, same `n=4`, acceptance differing by 0.22. Acceptance is a property
of how predictable the generated text is, so this says plainly that the two
runtimes are not asking the model to do the same thing. Without a
`reasoning_effort` directive the model reasons at a lower effort, and
lower-effort reasoning is more predictable, exactly the trend section 3
measures (`xhigh` 0.766 -> `medium` 0.820 -> `low` 0.839).

**What this costs you.** Ollama's speed here is partly bought by not applying
the model's template, which also means `reasoning_effort` cannot be set through
it, `low`/`medium`/`xhigh` live in the Jinja template Ollama discards. If you
want `medium` reasoning (section 3's best free win), you need `llama-server`
with `--jinja`.

**How to make this comparison honestly:** match the flags, match the template,
and report draft acceptance next to throughput. Acceptance is what exposes a
task that quietly changed.

---

## 7. Recommended configurations

**Fast interactive coding, the best all-round setting measured**

```bash
llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
  -ngl 999 -fa on --jinja -np 1 \
  -c 32768 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --spec-type draft-mtp --spec-draft-n-max 2
```
**136.7 tok/s, 17.0 GB, 100% quality**, measured as written, not extrapolated
from a neighbouring config.

Note the `n=2`. n=3 is the peak with an f16 cache, but **with q4_0 the two tie**
(136.75 vs 136.72 tok/s) while n=3 costs an extra 150 MB of draft context, so
n=2 is strictly better here. The optimal draft depth depends on the KV cache
type; do not carry n=3 over from section 2 by habit.

Add `--reasoning-effort medium` to cut the wait before answers by a third;
disable thinking entirely for simple tasks and throughput roughly doubles.

**Long-document work**

```bash
  -c 131072 --cache-type-k q8_0 --cache-type-v q8_0 \
  --spec-type draft-mtp --spec-draft-n-max 3
```
**121.8 tok/s, 21.7 GB**, ~77 s to ingest a full 128K prompt. Here n=3 is worth
it, but barely, 121.8 vs 120.6 at n=2, about 1%. Go to 256K only for batch
jobs.

**Do not bother with** `--spec-type ngram-mod` (0.97x on novel prompts, a net
loss), `-fa off` (costs 2.3 GB and 20% of prefill), or an f16 KV cache (slower
*and* larger than q4_0).

---

## 8. Measurement notes, why naive numbers lie

Six traps hit during this benchmark, each of which would have produced a
confidently wrong published figure:

1. **Hidden reasoning tokens.** Ollama's OpenAI-compatible endpoint generates
   thinking silently inside TTFT and streams only the answer, while still
   counting thinking in `completion_tokens`. Timing the visible stream gave
   **3477 tok/s**. Use the native `/api/chat` timings, or `completion_tokens /
   wall_s`.
2. **Cache replay.** n-gram drafting replays prompts it has already answered.
   Repeat-the-same-prompt benchmarks overstate it by up to 9x.
3. **VRAM baseline drift.** Desktop GPU usage moved 1.5-8 GB during the session
   (display sleep, apps opening). Subtracting a baseline captured once made MTP
   appear to *reduce* memory. Measure per-process VRAM, on Windows via the
   `GPU Process Memory` performance counter, since nvidia-smi returns N/A there.
4. **Power capping, not thermal throttling.** The 5090 sits at 590 W of a 600 W
   limit and downclocks to 2685 MHz from 3090 MHz. Cold-GPU "burst" numbers run
   7-8% above steady state. Never compare a config measured cold against one
   measured an hour in.
5. **Ollama is llama.cpp, with different defaults.** It launches
   `llama-server` as a child process, so "Ollama vs llama.cpp" compares
   configurations, not engines. Reading the child's command line settles what
   those configurations are:

   ```
   llama-server.exe --model <blob> -c 32768 -np 1 \
     --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-backend-sampling \
     --cache-type-k f16 --cache-type-v f16 --flash-attn on \
     -b 1024 -ub 1024 --context-shift --keep 4 \
     --no-jinja --chat-template chatml
   ```

   Ollama turns **MTP on by default at n=4**, adds `--spec-draft-backend-sampling`
   (a no-op, it is already the default), and overrides the batch defaults.
   Benchmarking it against a stock `llama-server` measures those flags, not the
   runtime.

6. **"Zero thinking tokens" can mean the parser moved, not the model.**
   Running llama-server with `--no-jinja --chat-template chatml` measured
   **2.13x** the stock baseline while reporting **0 thinking tokens**, which
   reads like an obvious story: the chatml template has none of Qwen3.8's
   reasoning scaffolding, so the model stopped reasoning and got faster.

   That story is wrong. The model reasons exactly as before, the raw response
   begins `<think>\nThe user wants a complete Python implementation of...`.
   Without `--jinja`, llama.cpp does not apply the model's reasoning format, so
   thinking is never split out into `reasoning_content`; it stays inline in
   `content`. A harness that counts reasoning by reading `reasoning_content`
   sees zero and concludes thinking was disabled.

   | config | decode tok/s | `reasoning_content` | actually reasoning? | accept |
   |---|---|---|---|---|
   | Ollama's MTP flags, model's own template | 120.6 | 106 tokens | yes | 0.591 |
   | Ollama's MTP flags, `chatml` template | 156.5 | **0** | **yes**, inline | 0.689 |

   The speedup is real but has a different cause: the chatml prompt carries no
   `reasoning_effort` directive, so the model reasons at a lower effort than the
   template's `xhigh` default, and lower-effort reasoning is more predictable, 
   acceptance rises 0.591 -> 0.689, matching the effort ladder in section 3.

   There is also a usability consequence worth knowing: under `--no-jinja`,
   `<think>` blocks are delivered to the client as ordinary answer text.

---

## 9. Open questions

Three questions from earlier drafts are now closed, each by direct
measurement rather than inference:

- ~~Is Ollama enabling MTP by default?~~ **Yes**, at `n=4`. The ~1.9x gap was
  MTP-on vs MTP-off. This reversed the earlier "Ollama cannot do MTP on CUDA"
  claim; the comparison is now a matched one (section 6a).
- ~~Is the n=3 peak thermally biased?~~ **No.** The reversed ladder still peaks
  at n=3 (section 2). Only n=5 was materially affected, by 3.9%.
- ~~Does `--spec-draft-backend-sampling` help?~~ **It is already the default**
  (`backend_sampling=1`), so there is nothing to enable, but disabling it
  costs 3.7-4.0% (section 4a).

Still open:

- **Do the quality tasks discriminate?** Everything scored 100%, including
  thinking-off and q4_0 cache. That proves these settings do not break basic
  competence; it does **not** prove reasoning is useless. A harder suite
  (the existing `hardbench.py`) would be needed to test that.
- **Why is cross-session drift ~6% when within-session repeatability is 0.2%?**
  Two runs of an identical config, back to back in one session, agree to 0.2%;
  the same config measured in a different session differed by 6%. Ambient
  temperature and background GPU load are the obvious suspects, but this was
  not isolated. Practical consequence: **only compare configs measured in the
  same session**, which is why every control here is a paired sweep rather than
  a comparison against a stored number.
- **Is Ollama's higher acceptance entirely the missing `reasoning_effort`
  directive?** Its 0.815 vs llama.cpp's 0.591 is consistent with lower-effort
  reasoning, but Ollama also ships different sampling defaults
  (`temperature 1, top_k 20, top_p 0.95, min_p 0` per `ollama show
  --parameters`), and sampling affects predictability too. The two were not
  separated.

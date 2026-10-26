# Status and reproduction

Benchmarking is **complete**. 45 configurations in `research/results.json`,
median of 3 repeats, warmup discarded. Findings are in `FINDINGS.md`; the
generated table is `research/results-summary.md` and the charts are in `blog/`.

The harness skips any config already cached, so re-running a stage is safe and
only fills gaps. Pass `--redo` to force re-measurement.

---

## Reproducing the whole sweep

```bash
cd research
python bench.py all          # every stage, ~4 hours
python summarize.py --md     # regenerate results-summary.md
python charts.py             # regenerate blog/*.png
python audit.py              # flag errored runs, wide spread, mixed timing
python verify_findings.py    # re-derive every number quoted in FINDINGS.md
```

Individual stages: `spec`, `specrev`, `bs`, `bsrev`, `kv`, `ctx`, `fa`,
`think`, `reasoning`, `runtime`, `recommend`.

---

## What each control run is for

Three stages exist only to falsify a result the obvious method would have got
wrong. They are the reason the numbers in `FINDINGS.md` can be trusted.

| stage | controls for | outcome |
|---|---|---|
| `specrev` | run order on a power-capped GPU: the ladder ran n=1 to n=5 on a heating card | n=3 still peaks; only n=5 moved (+3.9%) |
| `bsrev` | same, for the backend-sampling pair | gain unchanged (1.039x vs 1.037x) |
| `runtime` | comparing Ollama's defaults against llama.cpp's and calling it a runtime difference | two thirds of the gap was MTP being on by default |

**The rule these encode:** on a card that power-caps under sustained load
(this 5090 sits at 599 W of 600 W and drops from 3090 to 2700 MHz), whichever
config runs first is measured fastest. Any single-pass sweep is partly
measuring its own run order. Within one session the harness reproduces to
**0.2%**; across sessions the same config drifted **6%**, so compare only
within a session.

---

## Verifying the claims independently

These probe scripts each settle one question by direct observation. Several
overturned a conclusion this document previously stated as fact.

| script | question | needs GPU |
|---|---|---|
| `inspect_gguf.py` | does the GGUF really carry an MTP head? | no |
| `inspect_template.py` | what reasoning controls does the chat template define? | no |
| `probe_ollama_flags.py` | what flags does Ollama pass to its llama-server child? | yes |
| `probe_ollama_accept.py` | what draft acceptance does Ollama actually achieve? | yes |
| `probe_effort.py` | are `minimal`/`high`/`max` genuinely rejected? | yes |
| `probe_timing_parity.py` | is client-side timing comparable to server-reported? | yes |
| `probe_ollama_longctx.py` | reproduce the q8_0 long-prompt 500s | yes |
| `probe_stream.py` | which SSE fields carry reasoning vs answer? | yes |

**Run `probe_ollama_flags.py` before trusting any Ollama comparison.** Ollama
does not run stock llama.cpp, and the gap it appears to open is mostly its
defaults.

---

## Gotchas that cost real time here

- **The Ollama tray app restarts `ollama serve` within seconds of it being
  killed**, without your environment. A probe that sets `OLLAMA_DEBUG=1` will
  then silently attach to the tray's server and log nothing. Kill
  `ollama app.exe` too, and check port 11434 is closed before starting.
- **nvidia-smi cannot report per-process VRAM on Windows (WDDM).** Use the
  `GPU Process Memory(pid_N_*)\Dedicated Usage` performance counter. Whole-card
  readings drift by gigabytes as the desktop composites.
- **Ollama runs `llama-server` as a child**, so a PID lookup matching only
  `ollama*` finds a process holding ~0.3 MB.
- **Reasoning arrives as `reasoning_content`, not `content`**, but only under
  `--jinja`. Without it, thinking is inline in `content`, and a harness that
  counts `reasoning_content` reports zero thinking for a model that is
  reasoning normally.
- **`decode_tps` is not comparable across runtimes.** Ollama's comes from its
  own `eval_duration`, llama.cpp's from the client stream. Use
  `completion_tokens / wall_s` for cross-runtime numbers.
- Norton's `model_host.exe` holds 2-3 GB of VRAM across several processes.

---

## Scope: decided, not revisited

No quantization comparison (Q5_K_M / Q6_K / Q8_0), no multi-user `-np`
throughput, no `llama-bench`, no vLLM.

llama.cpp is the primary runtime throughout and is driven **directly** via
`tools\llama-server.exe` with explicit flags, e.g.:

```
llama-server.exe -m <gguf> -ngl 999 -c 32768 -np 1 --jinja --metrics \
  -fa on --cache-type-k f16 --cache-type-v f16 \
  --spec-type draft-mtp --spec-draft-n-max 3
```

Ollama appears only as a comparison, and it runs this same binary as a child
process, so there is no separate engine to test.

---

## Environment

- Model: Ollama blob `E:\AI\ollama\blobs\sha256-f5f1dd89...` (Qwen3.8-27B
  Q4_K_M, MTP head intact). No separate download needed.
- llama.cpp: `tools\llama-server.exe`, build b10448, CUDA 13.3 + bundled cudart.
- Ollama 0.32.13.
- GPU: RTX 5090 32 GB, power-capped at 600 W.

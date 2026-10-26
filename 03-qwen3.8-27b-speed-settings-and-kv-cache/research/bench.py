"""Qwen3.8-27B settings sweep driver.

Varies one axis at a time from a fixed baseline, rather than a full cartesian
product, so the results stay interpretable and the run stays finite.

Stages (run independently, results appended to results.json):

  spec     speculative decoding: off / draft-mtp n=1..5 / ngram-mod
  kv       KV cache type: f16 / q8_0 / q4_0
  ctx      context length: 4K .. 256K
  fa       flash attention on/off
  runtime  llama.cpp vs Ollama at matched settings

Usage:
    python bench.py spec
    python bench.py kv ctx
    python bench.py all
    python bench.py spec --redo      # ignore cached results for that stage
"""

import json
import os
import sys
import time
import traceback

import harness as H
import workloads as W

RESEARCH = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(RESEARCH, "results.json")
LOGS = os.path.join(RESEARCH, "logs")
os.makedirs(LOGS, exist_ok=True)

REPEATS = 3  # medians of 3, warmup discarded

# Baseline all single-axis sweeps vary from.
BASE = {
    "runtime": "llamacpp",
    "ctx": 32768,
    "kv": "f16",
    "flash_attn": "on",
    "spec": "none",
    "n_max": None,
    "parallel": 1,
    "think": True,  # Qwen3.8 reasons by default; this is the biggest lever.
    # Qwen3.8's chat template accepts exactly three levels and raises on
    # anything else: 'low', 'medium', 'xhigh' (its default). None = template
    # default. llama.cpp's --reasoning-effort also offers minimal/high/max,
    # which this model rejects outright.
    "effort": None,
    # Ollama passes --spec-draft-backend-sampling by default; upstream
    # llama.cpp defaults it off. None = leave unset (build default), which is
    # what every config measured before this axis existed used.
    "backend_sampling": None,
    # None = the model's own Jinja template. Ollama forces "chatml" instead,
    # discarding Qwen3.8's reasoning controls along with it.
    "chat_template": None,
    # Ollama also overrides llama.cpp's batch defaults (2048/512) with 1024/1024.
    "batch": None,
    "ubatch": None,
    # Free-form marker appended to the config id, so a control run (e.g. the
    # reversed ladder) does not collide with the original in results.json.
    "tag": None,
}

VALID_EFFORTS = ("low", "medium", "xhigh")


def cfg(**kw):
    c = dict(BASE)
    c.update(kw)
    if c.get("effort") and c["effort"] not in VALID_EFFORTS:
        raise ValueError(
            f"reasoning effort {c['effort']!r} is not supported by Qwen3.8; "
            f"choose one of {VALID_EFFORTS}"
        )
    parts = [
        c["runtime"],
        f"ctx{c['ctx'] // 1024}k",
        f"kv{c['kv']}",
        f"fa{c['flash_attn']}",
        f"spec{c['spec']}" + (f"n{c['n_max']}" if c.get("n_max") else ""),
        ("think" + (f"-{c['effort']}" if c.get("effort") else ""))
        if c["think"] else "nothink",
    ]
    # Appended only when set, so ids measured before these axes existed keep
    # matching their cached rows.
    if c.get("backend_sampling") is not None:
        parts.append("bs" + ("on" if c["backend_sampling"] else "off"))
    if c.get("batch") or c.get("ubatch"):
        parts.append(f"b{c.get('batch') or 0}u{c.get('ubatch') or 0}")
    if c.get("chat_template"):
        parts.append("tmpl" + c["chat_template"])
    if c.get("tag"):
        parts.append(c["tag"])
    c["id"] = "-".join(parts)
    return c


# ---------------------------------------------------------------- stages


def stage_spec():
    out = [cfg(spec="none")]
    for n in (1, 2, 3, 4, 5):
        out.append(cfg(spec="draft-mtp", n_max=n))
    out.append(cfg(spec="ngram-mod"))
    return out


def stage_kv():
    res = []
    for kv in ("f16", "q8_0", "q4_0"):
        res.append(cfg(kv=kv))                        # without MTP
        res.append(cfg(kv=kv, spec="draft-mtp", n_max=2))  # with MTP
    return res


def stage_ctx():
    res = []
    for k in (4, 8, 16, 32, 64, 128, 256):
        ctx = k * 1024
        # Long contexts need a quantised KV cache to fit alongside the weights.
        kv = "f16" if k <= 64 else "q8_0"
        res.append(cfg(ctx=ctx, kv=kv, spec="draft-mtp", n_max=2))
    return res


def stage_fa():
    return [cfg(flash_attn="on"), cfg(flash_attn="off")]


# What Ollama actually launches, read off its child llama-server's command
# line (probe_ollama_flags.py). It is not stock llama.cpp: MTP is on at n=4
# with backend sampling, and the batch defaults are overridden.
OLLAMA_EQUIV = dict(spec="draft-mtp", n_max=4, backend_sampling=True,
                    batch=1024, ubatch=1024)


def stage_runtime():
    """Is Ollama faster because it is Ollama, or because of its defaults?

    The earlier version of this stage compared Ollama against llama.cpp with
    speculation off, and Ollama duly came out ~1.9x ahead. That gap was the
    flags, not the runtime: Ollama silently enables the MTP draft head. Each kv
    setting now gets three rows -- Ollama, llama.cpp configured identically to
    Ollama, and llama.cpp on its own stock defaults -- so the runtime overhead
    and the defaults gap are separated.
    """
    res = []
    for kv in ("f16", "q8_0"):
        res.append(cfg(runtime="ollama", kv=kv))
        # Ollama's performance flags, but still the model's own template.
        res.append(cfg(runtime="llamacpp", kv=kv, **OLLAMA_EQUIV))
        # ...and its template too. Ollama forces chatml, which changes what the
        # model generates -- and with speculative decoding the decode rate
        # depends on how predictable the generated text is, so the template is
        # a performance variable, not just a formatting one.
        res.append(cfg(runtime="llamacpp", kv=kv, chat_template="chatml",
                       **OLLAMA_EQUIV))
        res.append(cfg(runtime="llamacpp", kv=kv))
    return res


def stage_bs():
    """--spec-draft-backend-sampling: on by default in Ollama, off upstream.

    Never measured before. Run at the ladder's peak and at Ollama's n=4 so the
    flag is isolated from the draft-length effect.
    """
    res = []
    for n in (3, 4):
        for bs in (True, False):
            res.append(cfg(spec="draft-mtp", n_max=n, backend_sampling=bs))
    return res


def stage_recommend():
    """Measure exactly the configs section 7 recommends.

    The KV and context stages swept at n=2, but n=3 is the ladder's peak, so
    the recommendations pair a q4_0/q8_0 cache with n=3 -- a combination that
    was never actually run. Quoting the n=2 numbers against an n=3 command line
    is the kind of small dishonesty that survives review; measure it instead.
    """
    return [
        cfg(kv="q4_0", spec="draft-mtp", n_max=3),
        cfg(ctx=131072, kv="q8_0", spec="draft-mtp", n_max=3),
    ]


def stage_bsrev():
    """Thermal control for stage_bs -- same pairs, flag order inverted.

    stage_bs measured every `on` config before its `off` twin, so `on` always
    got the cooler GPU. The measured gap (~3.8%) is smaller than the drift seen
    between two runs of an identical config, so ordering alone could produce
    it. Running `off` first inverts the gradient: if `on` still wins, the flag
    is real.
    """
    res = []
    for n in (3, 4):
        for bs in (False, True):
            res.append(cfg(spec="draft-mtp", n_max=n, backend_sampling=bs,
                           tag="rev"))
    return res


def stage_think():
    """Thinking on/off, with and without MTP.

    Qwen3.8 reasons by default and thinking tokens dominate short answers, so
    this axis moves end-to-end latency far more than any server flag.
    """
    res = []
    for think in (True, False):
        res.append(cfg(think=think))
        res.append(cfg(think=think, spec="draft-mtp", n_max=2))
    return res


def stage_specrev():
    """The MTP ladder again, reversed, as a thermal control.

    The forward ladder runs n=1 -> n=5, and this card is power-capped under
    sustained load (590W of 600W, clocks down from 3090 to 2685 MHz), so later
    configs are measured on a hotter, slower GPU. Within the n=4 config alone
    decode declined 119.5 -> 106.3 across repeats.

    Running the same ladder in reverse inverts that gradient. If n=3 still
    peaks, the optimum is real; if the peak moves, ordering was biasing it.
    The `tag` keeps these ids distinct from the forward run's.
    """
    out = [cfg(spec="draft-mtp", n_max=n, tag="rev") for n in (5, 4, 3, 2, 1)]
    out.append(cfg(spec="none", tag="rev"))
    return out


def stage_reasoning():
    """Qwen3.8's three reasoning-effort levels, plus thinking off entirely.

    The template injects a system instruction rather than enforcing a token
    cap, so the levels are a soft nudge. Measuring how many thinking tokens
    each level actually spends is the only way to know what it buys.
    """
    res = []
    for effort in ("xhigh", "medium", "low"):
        res.append(cfg(think=True, effort=effort, spec="draft-mtp", n_max=2))
    res.append(cfg(think=False, spec="draft-mtp", n_max=2))
    return res


STAGES = {
    "spec": stage_spec,
    "specrev": stage_specrev,
    "kv": stage_kv,
    "ctx": stage_ctx,
    "fa": stage_fa,
    "think": stage_think,
    "reasoning": stage_reasoning,
    "runtime": stage_runtime,
    "bs": stage_bs,
    "bsrev": stage_bsrev,
    "recommend": stage_recommend,
}


# ---------------------------------------------------------------- execution


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(data):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, RESULTS)


def build_suite(c):
    """Workloads for one config, sized to its context window."""
    suite = list(W.DECODE_WORKLOADS)
    suite += W.prefill_workloads(c["ctx"])
    suite += W.QUALITY_WORKLOADS
    suite += W.needle_suite(c["ctx"])
    return suite


def run_one_workload(url, model_name, task, c, baseline_vram, pid=None,
                     native=False):
    if native:
        return _run_ollama_workload(url, model_name, task, c, baseline_vram, pid)
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_tokens": task["max_tokens"],
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not c.get("think", True):
        # llama.cpp honours chat_template_kwargs; Ollama uses a top-level
        # `think` field. Sending both keeps the two runtimes comparable.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload["think"] = False
    else:
        if c.get("effort"):
            payload["chat_template_kwargs"] = {"reasoning_effort": c["effort"]}
        if task["kind"] in ("quality", "needle", "prefill"):
            # With reasoning on, thinking can consume the entire budget and
            # the answer never arrives, which would score as a wrong answer
            # rather than a truncation. Give correctness tasks room to finish.
            payload["max_tokens"] = task["max_tokens"] * 6
    sampler = H.GpuSampler(interval=0.25, pid=pid)
    sampler.start()
    try:
        text, timings = H.stream_chat(url, payload)
    finally:
        sampler.stop()

    rec = dict(timings)
    rec["task"] = task["id"]
    rec["kind"] = task["kind"]
    rec.update(sampler.summary(baseline_vram))

    check = task.get("check")
    rec["correct"] = bool(check(text)) if check else None
    rec["response_chars"] = len(text)
    rec["response_head"] = text[:200]

    # Energy efficiency: tokens generated per watt-hour.
    ct = rec.get("completion_tokens")
    pw = rec.get("power_mean_w")
    ds = rec.get("decode_s")
    if ct and pw and ds:
        wh = pw * (ds / 3600.0)
        rec["tokens_per_wh"] = round(ct / wh, 1) if wh > 0 else None
    return rec


def _run_ollama_workload(url, model_name, task, c, baseline_vram, pid):
    """Ollama path: native /api/chat with server-reported timings.

    num_ctx is passed per request rather than relying on the environment
    variable alone; without it Ollama falls back to a smaller window and
    returns HTTP 500 on prompts that exceed it.
    """
    max_tok = task["max_tokens"]
    if c.get("think", True) and task["kind"] in ("quality", "needle", "prefill"):
        max_tok *= 6
    options = {
        "num_ctx": c["ctx"],
        "temperature": 0,
        "num_predict": max_tok,
    }
    sampler = H.GpuSampler(interval=0.25, pid=pid)
    sampler.start()
    try:
        text, timings = H.stream_chat_ollama(
            url, model_name, task["prompt"], options, think=c.get("think", True)
        )
    finally:
        sampler.stop()

    rec = dict(timings)
    rec["task"] = task["id"]
    rec["kind"] = task["kind"]
    rec.update(sampler.summary(baseline_vram))

    check = task.get("check")
    rec["correct"] = bool(check(text)) if check else None
    rec["response_chars"] = len(text)
    rec["response_head"] = text[:200]

    ct, pw, ds = (
        rec.get("completion_tokens"),
        rec.get("power_mean_w"),
        rec.get("decode_s"),
    )
    if ct and pw and ds:
        wh = pw * (ds / 3600.0)
        rec["tokens_per_wh"] = round(ct / wh, 1) if wh > 0 else None
    return rec


def run_config(c, _unused_baseline=None):
    """Start the runtime, run the suite REPEATS times, tear down.

    The idle VRAM baseline is re-measured immediately before this config runs,
    not once for the whole sweep. Background desktop usage drifts by hundreds
    of megabytes over a multi-hour run, and a stale baseline silently corrupts
    every VRAM figure after it.
    """
    print(f"\n{'=' * 72}\n>>> {c['id']}\n{'=' * 72}", flush=True)

    baseline_vram = H.idle_vram(settle=2.0)
    print(f"    idle baseline now {baseline_vram:.0f} MiB", flush=True)

    if c["runtime"] == "ollama":
        rt = H.OllamaRuntime(c)
        model_name = H.OLLAMA_MODEL
    else:
        rt = H.LlamaServer(c, LOGS)
        model_name = "qwen3.8"

    result = {"config": c, "runs": [], "error": None,
              "baseline_vram_mib": baseline_vram}
    try:
        load_s = rt.start()
        result["load_seconds"] = load_s

        # Footprint of the loaded model + KV cache, before any request runs.
        # The per-process figure is authoritative; the whole-card one is kept
        # to show how much the rest of the desktop was holding at the time.
        time.sleep(2)
        pid = rt.pid
        resident = H.nvidia_query(["memory.used"])[0]
        result["server_pid"] = pid
        result["vram_resident_mib"] = round(resident - baseline_vram, 1)
        result["vram_resident_raw_mib"] = round(resident, 1)
        result["proc_vram_load_mib"] = H.process_vram_mib(pid) if pid else None
        print(
            f"    loaded in {load_s}s, process VRAM "
            f"{result['proc_vram_load_mib']} MiB "
            f"(whole card {resident:.0f} MiB)",
            flush=True,
        )

        suite = build_suite(c)

        # Warmup, discarded: settles clocks and fills any prompt cache.
        try:
            run_one_workload(
                rt.chat_url(), model_name, W.DECODE_WORKLOADS[0], c,
                baseline_vram, pid, rt.native
            )
        except Exception as e:
            print(f"    warmup failed: {e}", flush=True)

        for rep in range(REPEATS):
            for task in suite:
                # Prefill/needle prompts are expensive; run them once, not 3x.
                if task["kind"] in ("prefill", "needle") and rep > 0:
                    continue
                try:
                    rec = run_one_workload(
                        rt.chat_url(), model_name, task, c,
                        baseline_vram, pid, rt.native
                    )
                except Exception as e:
                    rec = {"task": task["id"], "kind": task["kind"], "error": str(e)}
                rec["rep"] = rep
                result["runs"].append(rec)
                print(
                    f"    [{rep}] {rec['task']:<18} "
                    f"ttft={rec.get('ttft_s')} "
                    f"dec={rec.get('decode_tps')}tps "
                    f"pre={rec.get('prefill_tps')}tps "
                    f"vram={rec.get('vram_peak_mib')}MiB "
                    f"ok={rec.get('correct')}"
                    + (f" ERR={rec['error'][:60]}" if rec.get("error") else ""),
                    flush=True,
                )

        if c["runtime"] == "llamacpp":
            result["draft_stats"] = H.parse_draft_acceptance(
                rt.log_text(), rt.metrics()
            )
            print(f"    draft: {result['draft_stats']}", flush=True)

    except Exception as e:
        result["error"] = f"{e}\n{traceback.format_exc()}"
        print(f"    !! FAILED: {e}", flush=True)
    finally:
        rt.stop()
        # Never start the next config until the card is actually free, or the
        # runtime-vs-runtime comparison is not like-for-like.
        released = H.wait_vram_release(baseline_vram)
        result["vram_after_release_mib"] = released
        print(f"    GPU released, VRAM back to {released:.0f} MiB", flush=True)
    return result


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    redo = "--redo" in sys.argv
    names = list(STAGES) if (not args or args == ["all"]) else args
    for n in names:
        if n not in STAGES:
            print(f"unknown stage: {n}. choose from {list(STAGES)} or 'all'")
            return

    H._kill_stragglers("llama-server.exe")
    baseline = H.idle_vram()
    print(f"Idle VRAM at sweep start = {baseline:.0f} MiB")
    print("(re-measured before every config; background usage drifts)\n")

    data = load_results()
    data.setdefault("_meta", {})["baseline_vram_mib"] = baseline
    data["_meta"]["gpu"] = "RTX 5090 32GB"
    data["_meta"]["model"] = "Qwen3.8-27B Q4_K_M (GGUF, MTP head present)"
    data["_meta"]["llama_build"] = "b10448"
    data["_meta"]["repeats"] = REPEATS

    for name in names:
        configs = STAGES[name]()
        print(f"\n### stage '{name}': {len(configs)} configs")
        for c in configs:
            key = c["id"]
            if key in data and not redo and not data[key].get("error"):
                print(f"  skip (cached): {key}")
                continue
            data[key] = run_config(c)
            save_results(data)
    print(f"\nSaved {RESULTS}")


if __name__ == "__main__":
    main()

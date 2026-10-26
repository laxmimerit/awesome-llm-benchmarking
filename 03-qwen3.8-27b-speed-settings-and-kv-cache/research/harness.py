"""Core benchmark engine for the Qwen3.8-27B settings sweep.

Provides:
  * Runtime wrappers that start/stop llama-server and Ollama with a given config
  * A GPU sampler thread (VRAM + power draw) built on nvidia-smi
  * A streaming OpenAI-compatible client that timestamps every token, so we can
    separate time-to-first-token from steady-state decode speed

Only uses the Python standard library, matching the other harnesses in this
project.
"""

import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
LLAMA_SERVER = os.path.join(TOOLS, "llama-server.exe")

# The Qwen3.8-27B Q4_K_M GGUF, reused straight out of the Ollama blob store so
# we do not keep a second 15.7 GB copy on disk.
MODEL_GGUF = (
    r"E:\AI\ollama\blobs"
    r"\sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"
)
OLLAMA_MODEL = "qwen3.8:latest"

LLAMA_PORT = 8099
OLLAMA_URL = "http://127.0.0.1:11434"


# ---------------------------------------------------------------- GPU sampling


def nvidia_query(fields):
    """One-shot nvidia-smi query. Returns a list of floats."""
    out = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    return [float(x.strip()) for x in out.split(",")]


def process_vram_mib(pid):
    """Dedicated GPU memory held by one process, in MiB.

    nvidia-smi cannot report per-process VRAM under Windows' WDDM driver
    model (it returns N/A), but the Windows performance counter can. This is
    the only way to get a model's footprint that is not polluted by whatever
    the desktop, browsers, and background apps happen to be holding.
    """
    ps = (
        f"$s=(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage' "
        "-ErrorAction SilentlyContinue).CounterSamples; "
        "if($s){ ($s | Measure-Object CookedValue -Sum).Sum } else { 0 }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True,
            text=True,
            timeout=25,
        ).stdout.strip()
        return round(float(out) / (1024 * 1024), 1) if out else None
    except Exception:
        return None


class GpuSampler(threading.Thread):
    """Background sampler for VRAM used and board power draw.

    Samples whole-card memory and power quickly, and (when given a pid) the
    process's own dedicated VRAM on a slower cadence, since the Windows
    counter query costs far more than an nvidia-smi field read.
    """

    def __init__(self, interval=0.25, pid=None, proc_every=6):
        super().__init__(daemon=True)
        self.interval = interval
        self.pid = pid
        self.proc_every = proc_every
        self._stop = threading.Event()
        self.mem = []
        self.power = []
        self.proc_mem = []
        # Clock and temperature make throttling visible in the results. This
        # card is power-capped under sustained load, so a config measured on a
        # cold GPU is not comparable to one measured an hour in.
        self.clock = []
        self.temp = []

    def run(self):
        i = 0
        while not self._stop.is_set():
            try:
                mem, pwr, clk, temp = nvidia_query(
                    ["memory.used", "power.draw",
                     "clocks.current.graphics", "temperature.gpu"]
                )
                self.mem.append(mem)
                self.power.append(pwr)
                self.clock.append(clk)
                self.temp.append(temp)
            except Exception:
                pass
            if self.pid and i % self.proc_every == 0:
                v = process_vram_mib(self.pid)
                if v:
                    self.proc_mem.append(v)
            i += 1
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)

    def summary(self, baseline_mem=0.0):
        """Peak/mean VRAM and power stats.

        `proc_vram_peak_mib` is the authoritative model footprint: it is the
        server process's own memory, so a browser waking up mid-run cannot
        inflate it. The baseline-subtracted whole-card figures are kept for
        comparison and to expose contention.
        """
        if not self.mem:
            return {}
        out = {
            "vram_peak_mib": round(max(self.mem) - baseline_mem, 1),
            "vram_mean_mib": round(statistics.mean(self.mem) - baseline_mem, 1),
            "vram_peak_raw_mib": round(max(self.mem), 1),
            "power_mean_w": round(statistics.mean(self.power), 1) if self.power else None,
            "power_peak_w": round(max(self.power), 1) if self.power else None,
            "samples": len(self.mem),
        }
        if self.clock:
            out["clock_mean_mhz"] = round(statistics.mean(self.clock))
            out["clock_min_mhz"] = round(min(self.clock))
        if self.temp:
            out["temp_mean_c"] = round(statistics.mean(self.temp))
            out["temp_peak_c"] = round(max(self.temp))
        if self.proc_mem:
            out["proc_vram_peak_mib"] = round(max(self.proc_mem), 1)
            out["proc_vram_samples"] = len(self.proc_mem)
        return out


def idle_vram(settle=3.0):
    """VRAM in use with no benchmark model loaded (desktop, browsers, etc.)."""
    time.sleep(settle)
    vals = []
    for _ in range(6):
        vals.append(nvidia_query(["memory.used"])[0])
        time.sleep(0.3)
    return statistics.median(vals)


SERVER_PROCESSES = (
    "llama-server.exe",
    "ollama.exe",
    "ollama app.exe",
    "ollama_llama_server.exe",
    "ollama runner.exe",
)

# The model weights alone are ~16.6 GB, so anything below this means no
# benchmark model is resident, whatever the desktop happens to be using.
MODEL_GONE_MIB = 12000


def _server_processes_alive(min_vram_mib=500):
    """Server processes that are actually holding GPU memory.

    Presence alone is not contention: Ollama installs a tray app that sits
    resident with no model loaded and no VRAM. Waiting for it to exit would
    stall every teardown for the full timeout, so judge by memory held.
    """
    out = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True
    ).stdout
    holding = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        name = parts[0].lower()
        if not any(p.lower() == name for p in SERVER_PROCESSES):
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if (process_vram_mib(pid) or 0) >= min_vram_mib:
            holding.append(f"{parts[0]}({pid})")
    return holding


def wait_vram_release(baseline_mib=None, timeout=180):
    """Block until the benchmark model has really left the GPU.

    Correctness here is "no model is resident", not "VRAM equals some number".
    Desktop VRAM drifts by gigabytes as the display sleeps and wakes, so an
    absolute baseline comparison produces false alarms in both directions.
    Instead: confirm no server process survives, then wait for the reading to
    stop moving.
    """
    t0 = time.time()
    alive = _server_processes_alive()
    while alive and time.time() - t0 < timeout:
        time.sleep(1.0)
        alive = _server_processes_alive()
    if alive:
        print(
            f"    !! WARNING: server processes still running: {alive} - "
            "next config may be contaminated",
            flush=True,
        )

    # Settle: wait until successive readings agree, so we do not sample during
    # the allocator's teardown.
    last, stable = None, 0
    while time.time() - t0 < timeout:
        try:
            used = nvidia_query(["memory.used"])[0]
        except Exception:
            time.sleep(1.0)
            continue
        if last is not None and abs(used - last) < 150:
            stable += 1
        else:
            stable = 0
        last = used
        if stable >= 2 and used < MODEL_GONE_MIB:
            time.sleep(1.5)
            return round(used, 1)
        time.sleep(1.5)

    used = nvidia_query(["memory.used"])[0]
    if used >= MODEL_GONE_MIB:
        print(
            f"    !! WARNING: {used:.0f} MiB still in use after {timeout}s - "
            "a model may still be resident; next config may be skewed",
            flush=True,
        )
    return round(used, 1)


# ---------------------------------------------------------------- HTTP helpers


def http_json(url, payload=None, timeout=60, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method or ("POST" if data else "GET"),
    )
    with urllib.request.urlopen(req, timeout=timeout) as f:
        body = f.read().decode("utf-8", "replace")
    return json.loads(body) if body.strip() else {}


def wait_healthy(url, timeout=900):
    """Poll a health endpoint until the model is loaded. Returns load seconds."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as f:
                if f.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server not healthy after {timeout}s: {url}")


# ---------------------------------------------------------------- streaming


def stream_chat_ollama(url, model, prompt, options, think=True, timeout=1800):
    """Stream from Ollama's native /api/chat and use its own timing fields.

    Ollama's OpenAI-compatible endpoint does not emit reasoning as streamed
    deltas: thinking tokens are generated silently and only the final answer
    arrives as chunks, while `completion_tokens` still counts the thinking.
    Timing the visible stream therefore divides all the tokens by only the
    answer's duration and reports absurd throughput.

    The native endpoint reports `eval_count` / `eval_duration` measured by the
    server itself, which is exact and directly comparable to what we compute
    for llama.cpp.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": options,
        "think": bool(think),
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    t_first = t_first_answer = None
    content, thinking = [], []
    final = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = obj.get("message") or {}
            piece = msg.get("content")
            think_piece = msg.get("thinking")
            if piece or think_piece:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                if think_piece:
                    thinking.append(think_piece)
                if piece:
                    if t_first_answer is None:
                        t_first_answer = now
                    content.append(piece)
            if obj.get("done"):
                final = obj

    t_end = time.perf_counter()
    text = "".join(content)

    eval_count = final.get("eval_count") or 0
    eval_ns = final.get("eval_duration") or 0
    p_count = final.get("prompt_eval_count") or 0
    p_ns = final.get("prompt_eval_duration") or 0

    timings = {
        "wall_s": round(t_end - t_start, 4),
        "ttft_s": round(t_first - t_start, 4) if t_first else None,
        "ttfa_s": round(t_first_answer - t_start, 4) if t_first_answer else None,
        "prompt_tokens": p_count or None,
        "completion_tokens": eval_count or None,
        "thinking_chars": len("".join(thinking)),
        # Server-reported, so thinking time is included where it belongs.
        "decode_s": round(eval_ns / 1e9, 4) if eval_ns else None,
        "decode_tps": (
            round(eval_count / (eval_ns / 1e9), 2) if eval_ns and eval_count else None
        ),
        "prefill_tps": (
            round(p_count / (p_ns / 1e9), 2) if p_ns and p_count else None
        ),
        "timing_source": "ollama-native",
    }
    if not eval_ns:
        timings["error"] = "ollama returned no eval_duration"
    return text, timings


def stream_chat(url, payload, timeout=1800):
    """POST a streaming chat completion, timestamping each token.

    Returns (text, timings) where timings carries TTFT and the per-token
    inter-arrival deltas needed for a true steady-state decode rate.
    """
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    t_first = None          # first token of any kind (what the user perceives)
    t_first_answer = None   # first token of the actual answer, post-thinking
    stamps = []
    chunks = []
    think_chunks = []
    usage = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta") or {}
                # Qwen3.8 is a reasoning model: thinking arrives as
                # `reasoning_content`, the answer as `content`. Both are real
                # generated tokens and both must count toward decode speed.
                think = delta.get("reasoning_content")
                piece = delta.get("content")
                if not think and not piece:
                    continue
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                stamps.append(now)
                if think:
                    think_chunks.append(think)
                if piece:
                    if t_first_answer is None:
                        t_first_answer = now
                    chunks.append(piece)

    t_end = time.perf_counter()
    text = "".join(chunks)
    thinking = "".join(think_chunks)

    if t_first is None:  # nothing generated at all
        return text, {"error": "no tokens generated", "wall_s": t_end - t_start}

    # Steady-state decode: exclude the prefill segment entirely.
    decode_s = stamps[-1] - t_first
    n_chunks = len(stamps)
    timings = {
        "ttft_s": round(t_first - t_start, 4),
        "ttfa_s": (
            round(t_first_answer - t_start, 4) if t_first_answer else None
        ),
        "wall_s": round(t_end - t_start, 4),
        "decode_s": round(decode_s, 4),
        "stream_chunks": n_chunks,
        "thinking_chunks": len(think_chunks),
        "thinking_chars": len(thinking),
        # tok/s measured from the stream itself (chunk-based, runtime-agnostic)
        "decode_tps_stream": (
            round((n_chunks - 1) / decode_s, 2) if decode_s > 0 else None
        ),
    }

    # Prefer server-reported token counts when available; they are exact.
    if usage:
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        timings["prompt_tokens"] = pt
        timings["completion_tokens"] = ct
        if ct and decode_s > 0:
            timings["decode_tps"] = round((ct - 1) / decode_s, 2)
        if pt and timings["ttft_s"] > 0:
            timings["prefill_tps"] = round(pt / timings["ttft_s"], 2)
    return text, timings


# ---------------------------------------------------------------- llama.cpp


class LlamaServer:
    """Manages one llama-server process for a given configuration."""

    native = False  # speaks the OpenAI-compatible API

    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        self.log_dir = log_dir
        self.proc = None
        self.log_path = os.path.join(log_dir, f"llama-{cfg['id']}.log")
        self.load_s = None

    def build_args(self):
        c = self.cfg
        args = [
            LLAMA_SERVER,
            "-m", MODEL_GGUF,
            "-ngl", str(c.get("ngl", 999)),
            "-c", str(c["ctx"]),
            "-np", str(c.get("parallel", 1)),
            "--host", "127.0.0.1",
            "--port", str(LLAMA_PORT),
            "--metrics",
            "-fa", c.get("flash_attn", "on"),
            "--cache-type-k", c.get("kv", "f16"),
            "--cache-type-v", c.get("kv", "f16"),
        ]
        # Default is the model's own Jinja template, which is what exposes
        # Qwen3.8's reasoning_effort controls. Ollama instead forces the
        # built-in chatml template, so this reproduces that.
        if c.get("chat_template"):
            args += ["--no-jinja", "--chat-template", c["chat_template"]]
        else:
            args += ["--jinja"]
        spec = c.get("spec")
        if spec and spec != "none":
            args += ["--spec-type", spec]
            if c.get("n_max") is not None:
                args += ["--spec-draft-n-max", str(c["n_max"])]
            if c.get("draft_kv"):
                args += [
                    "--spec-draft-type-k", c["draft_kv"],
                    "--spec-draft-type-v", c["draft_kv"],
                ]
            # Ollama turns this on by default; upstream llama.cpp defaults it off.
            # Leaving it unset means "use the build default", which is what every
            # config measured before this flag existed in the harness did.
            if c.get("backend_sampling") is not None:
                args += ["--spec-draft-backend-sampling" if c["backend_sampling"]
                         else "--no-spec-draft-backend-sampling"]
        if c.get("threads"):
            args += ["-t", str(c["threads"])]
        if c.get("batch"):
            args += ["-b", str(c["batch"])]
        if c.get("ubatch"):
            args += ["-ub", str(c["ubatch"])]
        return args

    def start(self, timeout=1200):
        args = self.build_args()
        self.logf = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.logf.write("CMD: " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n\n")
        self.logf.flush()
        t0 = time.time()
        self.proc = subprocess.Popen(
            args,
            stdout=self.logf,
            stderr=subprocess.STDOUT,
            cwd=TOOLS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        wait_healthy(f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=timeout)
        self.load_s = round(time.time() - t0, 2)
        return self.load_s

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def chat_url(self):
        return f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions"

    def metrics(self):
        """Scrape llama-server's Prometheus endpoint for draft acceptance."""
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{LLAMA_PORT}/metrics", timeout=10
            ) as f:
                return f.read().decode("utf-8", "replace")
        except Exception:
            return ""

    def log_text(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception:
            return ""

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=30)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        try:
            self.logf.close()
        except Exception:
            pass
        _kill_stragglers("llama-server.exe")
        time.sleep(2)


def _kill_stragglers(image):
    subprocess.run(
        ["taskkill", "/F", "/IM", image],
        capture_output=True,
        text=True,
    )


# --------------------------------------------------- MTP acceptance parsing


# Actual llama-server b10448 format, confirmed from a live run:
#   slot print_timing: id  0 | task 0 | draft acceptance = 0.71429
#       (  300 accepted /   420 generated), mean len =  2.43
DRAFT_LINE = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\)"
    r"(?:\s*,\s*mean len\s*=\s*([0-9.]+))?"
)


def parse_draft_acceptance(log_text, metrics_text=""):
    """Extract speculative-decoding acceptance from llama-server output.

    Aggregates across every request the server handled, so the reported rate
    reflects the whole workload mix rather than a single lucky prompt.
    """
    out = {}
    accepted = generated = 0
    rates, lens = [], []
    for m in DRAFT_LINE.finditer(log_text):
        rates.append(float(m.group(1)))
        accepted += int(m.group(2))
        generated += int(m.group(3))
        if m.group(4):
            lens.append(float(m.group(4)))

    if generated:
        out["draft_accepted"] = accepted
        out["draft_generated"] = generated
        out["accept_rate"] = round(accepted / generated, 4)
        out["n_requests"] = len(rates)
        if lens:
            out["mean_accepted_len"] = round(statistics.mean(lens), 3)
        return out

    # Fall back to the Prometheus endpoint if the log format ever changes.
    for key, label in (
        ("llamacpp:n_draft_total", "draft_generated"),
        ("llamacpp:n_draft_accepted", "draft_accepted"),
    ):
        m = re.search(rf"^{re.escape(key)}\s+([0-9.eE+]+)", metrics_text, re.M)
        if m:
            out[label] = float(m.group(1))
    if out.get("draft_generated"):
        out["accept_rate"] = round(out["draft_accepted"] / out["draft_generated"], 4)
    return out


def parse_draft_acceptance_tail(log_text, since_offset=0):
    """Acceptance for just the requests logged after `since_offset` chars."""
    return parse_draft_acceptance(log_text[since_offset:])


# ---------------------------------------------------------------- Ollama


class OllamaRuntime:
    """Restarts the Ollama service with the env knobs a config needs.

    Ollama exposes KV cache type / flash attention only as process-level
    environment variables, so changing them means restarting the service.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.load_s = None

    @staticmethod
    def _stop_service():
        _kill_stragglers("ollama.exe")
        _kill_stragglers("ollama app.exe")
        time.sleep(3)

    def start(self, timeout=600):
        self._stop_service()
        env = dict(os.environ)
        env["OLLAMA_KV_CACHE_TYPE"] = self.cfg.get("kv", "f16")
        env["OLLAMA_FLASH_ATTENTION"] = (
            "1" if self.cfg.get("flash_attn", "on") == "on" else "0"
        )
        env["OLLAMA_NUM_PARALLEL"] = str(self.cfg.get("parallel", 1))
        env["OLLAMA_CONTEXT_LENGTH"] = str(self.cfg["ctx"])
        env["OLLAMA_KEEP_ALIVE"] = "30m"
        exe = shutil.which("ollama") or os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")
        self.proc = subprocess.Popen(
            [exe, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        t0 = time.time()
        wait_healthy(f"{OLLAMA_URL}/api/version", timeout=120)

        # Force the model to load now so load time is measured, not amortised.
        http_json(
            f"{OLLAMA_URL}/api/generate",
            {
                "model": OLLAMA_MODEL,
                "prompt": "hi",
                "stream": False,
                "options": {"num_ctx": self.cfg["ctx"], "num_predict": 1},
            },
            timeout=timeout,
        )
        self.load_s = round(time.time() - t0, 2)
        return self.load_s

    native = True  # use /api/chat, not the OpenAI-compatible shim

    def chat_url(self):
        return f"{OLLAMA_URL}/api/chat"

    # Ollama does not implement inference itself: it launches llama.cpp's
    # llama-server as a child and talks to it. The `ollama` process is only an
    # API front-end and holds no GPU memory, so the model footprint has to be
    # read from the child. Names to consider, widest first.
    RUNNER_HINTS = ("llama-server", "ollama_llama_server", "ollama runner", "ollama")

    @property
    def pid(self):
        """PID of the process actually holding the model."""
        out = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True
        ).stdout
        best, best_mem = None, 0
        for line in out.splitlines():
            parts = [p.strip('"') for p in line.split('","')]
            if len(parts) < 2:
                continue
            name = parts[0].lower()
            if not any(h in name for h in self.RUNNER_HINTS):
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            mem = process_vram_mib(pid) or 0
            if mem > best_mem:
                best, best_mem = pid, mem
        return best

    def stop(self):
        # Ask Ollama to evict the model first; killing the process alone can
        # leave the GPU allocation held briefly by the runner subprocess.
        try:
            http_json(
                f"{OLLAMA_URL}/api/generate",
                {"model": OLLAMA_MODEL, "keep_alive": 0},
                timeout=60,
            )
        except Exception:
            pass
        time.sleep(2)
        self._stop_service()
        _kill_stragglers("ollama_llama_server.exe")
        _kill_stragglers("ollama runner.exe")
        time.sleep(2)

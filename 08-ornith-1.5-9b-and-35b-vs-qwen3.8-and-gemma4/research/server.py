"""Launch and control llama-server for one benchmark configuration.

Every model in this project goes through this one code path, so a speed
difference is never a difference in how the server was started.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from models import REGISTRY, SERVER

HOST = "127.0.0.1"
PORT = 8099
BASE = f"http://{HOST}:{PORT}"
LOGDIR = Path(__file__).resolve().parent.parent / "logs"


def gpu_mem_used_mb():
    """Total VRAM in use on device 0, in MiB, across all processes."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def gpu_state():
    q = ("memory.used,memory.total,temperature.gpu,power.draw,"
         "clocks.sm,clocks.max.sm,utilization.gpu")
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    vals = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
    keys = ["mem_used_mb", "mem_total_mb", "temp_c", "power_w",
            "clock_sm_mhz", "clock_sm_max_mhz", "util_pct"]
    d = {}
    for k, v in zip(keys, vals):
        try:
            d[k] = float(v)
        except ValueError:
            d[k] = None
    return d


def throttle_reason():
    out = subprocess.run(
        ["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True
    )
    active = []
    for line in out.stdout.splitlines():
        if ":" in line and "Active" in line:
            name = line.split(":")[0].strip()
            if line.split(":")[1].strip() == "Active":
                active.append(name)
    return active


class Server:
    def __init__(self, model_key, ctx=8192, ngl=999, flash_attn="on",
                 parallel=1, cache_type_k=None, cache_type_v=None,
                 batch=None, ubatch=None, extra=None, tag="",
                 spec_type=None, spec_n_max=None):
        self.model_key = model_key
        self.model = REGISTRY[model_key]
        self.ctx = ctx
        self.ngl = ngl
        self.flash_attn = flash_attn
        self.parallel = parallel
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.batch = batch
        self.ubatch = ubatch
        self.extra = extra or []
        # Multi-token prediction. Both Ornith GGUFs ship a `nextn` block that
        # llama.cpp ignores unless this is set, and neither model card mentions
        # it, so it is an explicit axis rather than a hidden default.
        self.spec_type = spec_type
        self.spec_n_max = spec_n_max
        self.tag = tag or model_key
        self.proc = None
        self.baseline_mb = None
        self.loaded_mb = None
        self.log_path = None

    def cmd(self):
        c = [
            str(SERVER),
            "-m", str(self.model["path"]),
            "--host", HOST, "--port", str(PORT),
            "-c", str(self.ctx),
            "-ngl", str(self.ngl),
            "-fa", self.flash_attn,
            "-np", str(self.parallel),
            "--jinja",
            "--no-warmup",
            "--metrics",
        ]
        if self.cache_type_k:
            c += ["-ctk", self.cache_type_k]
        if self.cache_type_v:
            c += ["-ctv", self.cache_type_v]
        if self.batch:
            c += ["-b", str(self.batch)]
        if self.ubatch:
            c += ["-ub", str(self.ubatch)]
        if self.spec_type:
            c += ["--spec-type", self.spec_type]
        if self.spec_n_max:
            c += ["--spec-draft-n-max", str(self.spec_n_max)]
        c += self.extra
        return c

    def start(self, timeout=600):
        LOGDIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOGDIR / f"server_{self.tag}.log"
        self.baseline_mb = gpu_mem_used_mb()

        env = dict(os.environ)
        # llama.cpp needs its DLLs; run with cwd set to the tools dir.
        self.proc = subprocess.Popen(
            self.cmd(),
            stdout=open(self.log_path, "w", encoding="utf-8", errors="replace"),
            stderr=subprocess.STDOUT,
            cwd=str(SERVER.parent),
            env=env,
        )

        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = Path(self.log_path).read_text(errors="replace")[-2000:]
                raise RuntimeError(
                    f"llama-server exited {self.proc.returncode}\n{tail}"
                )
            try:
                with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                    if r.status == 200:
                        self.load_s = time.time() - t0
                        time.sleep(1.0)
                        self.loaded_mb = gpu_mem_used_mb()
                        return self
            except Exception:
                time.sleep(1.0)
        self.stop()
        raise TimeoutError(f"server for {self.tag} did not become healthy in {timeout}s")

    def props(self):
        with urllib.request.urlopen(f"{BASE}/props", timeout=10) as r:
            return json.load(r)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
        # wait for VRAM to actually come back
        t0 = time.time()
        while time.time() - t0 < 60:
            if gpu_mem_used_mb() <= (self.baseline_mb or 0) + 500:
                break
            time.sleep(1.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def completion(prompt, n_predict=256, sampling=None, stream=False,
               timeout=1200, **kw):
    """Call llama.cpp's native /completion endpoint.

    Native endpoint is used rather than /v1/chat/completions because it returns
    a `timings` block measured inside the server, which is comparable across
    models in a way that client-side token counting is not.
    """
    body = {"prompt": prompt, "n_predict": n_predict, "stream": stream,
            "cache_prompt": False}
    if sampling:
        body.update(sampling)
    body.update(kw)
    req = urllib.request.Request(
        f"{BASE}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall_s"] = time.perf_counter() - t0
    return out


def chat(messages, n_predict=1024, sampling=None, timeout=2400, **kw):
    """OpenAI-compatible chat endpoint, used where the chat template matters."""
    body = {"messages": messages, "max_tokens": n_predict, "stream": False}
    if sampling:
        body.update(sampling)
    body.update(kw)
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    out["_wall_s"] = time.perf_counter() - t0
    return out


def tokenize(text):
    req = urllib.request.Request(
        f"{BASE}/tokenize",
        data=json.dumps({"content": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["tokens"]

"""Find out what flags Ollama actually passes to its llama-server child.

Ollama measured ~1.9x faster than our own llama-server on the same GGUF, and
landed suspiciously close to our MTP n=3 figure. Since Ollama runs llama.cpp
underneath, the explanation should be visible in the child process's command
line. This starts Ollama, loads the model, and prints that command line.

Run with the benchmark sweep stopped -- it needs the GPU.
"""

import json
import os
import shutil
import subprocess
import time
import urllib.request

import harness as H

CMDLINE_PS = r"""
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match 'llama|ollama' } |
  ForEach-Object { "$($_.ProcessId)`t$($_.Name)`t$($_.CommandLine)" }
"""


def cmdlines():
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", CMDLINE_PS],
        capture_output=True, text=True, timeout=60,
    ).stdout
    return [l for l in out.splitlines() if l.strip()]


print("stopping any running servers...")
H._kill_stragglers("llama-server.exe")
H.OllamaRuntime._stop_service()
time.sleep(3)

env = dict(os.environ)
env["OLLAMA_KV_CACHE_TYPE"] = "f16"
env["OLLAMA_FLASH_ATTENTION"] = "1"
env["OLLAMA_NUM_PARALLEL"] = "1"
env["OLLAMA_CONTEXT_LENGTH"] = "32768"
env["OLLAMA_KEEP_ALIVE"] = "5m"
env["OLLAMA_DEBUG"] = "1"

exe = shutil.which("ollama") or os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")
log = open("ollama-serve.log", "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen(
    [exe, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
)
H.wait_healthy(f"{H.OLLAMA_URL}/api/version", timeout=120)
print("ollama serve up; loading model...")

H.http_json(
    f"{H.OLLAMA_URL}/api/generate",
    {"model": H.OLLAMA_MODEL, "prompt": "hi", "stream": False,
     "options": {"num_ctx": 32768, "num_predict": 8}},
    timeout=600,
)
time.sleep(2)

print("\n=== process command lines ===")
for line in cmdlines():
    pid, name, *rest = line.split("\t")
    cmd = rest[0] if rest else ""
    print(f"\n[{name} pid {pid}]")
    print(cmd)

print("\n=== speculative / draft mentions in the runner command ===")
for line in cmdlines():
    low = line.lower()
    for kw in ("spec", "draft", "mtp", "ctx-size", "batch", "flash", "cache-type", "parallel"):
        if kw in low:
            print(f"  contains '{kw}'")
            break

print("\n=== ollama serve log lines mentioning spec/draft/mtp ===")
log.flush()
try:
    with open("ollama-serve.log", encoding="utf-8", errors="replace") as f:
        for line in f:
            low = line.lower()
            if any(k in low for k in ("spec", "draft", "mtp", "nextn")):
                print("   ", line.strip()[:200])
except Exception as e:
    print("  (log unreadable:", e, ")")

print("\n=== /api/ps ===")
try:
    print(json.dumps(H.http_json(f"{H.OLLAMA_URL}/api/ps"), indent=2)[:800])
except Exception as e:
    print(e)

H.OllamaRuntime._stop_service()
print("\ndone; ollama stopped")

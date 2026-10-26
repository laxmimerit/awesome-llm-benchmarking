"""Ollama returns HTTP 500 on long prompts with a q8_0 KV cache. Reproduce it.

The f16 run of the same config completed all 26 workloads; the q8_0 run failed
exactly the two longest (prefill_75pct at ~24K tokens, needle_d90). Either
OLLAMA_KV_CACHE_TYPE=q8_0 is not being applied the way num_ctx is, or the
quantised cache path has a limit the f16 path does not.

Sends prompts of increasing length and reports where it breaks, then prints the
server-side error.
"""

import os
import re
import shutil
import socket
import subprocess
import time

import harness as H
import workloads as W

exe = shutil.which("ollama") or \
    os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")


def port_open(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


H._kill_stragglers("llama-server.exe")
H.OllamaRuntime._stop_service()
H._kill_stragglers("ollama app.exe")
H._kill_stragglers("ollama.exe")
time.sleep(3)
if port_open(11434):
    raise SystemExit("port 11434 still occupied")

env = dict(os.environ)
env.update({
    "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_CONTEXT_LENGTH": "32768",
    "OLLAMA_KEEP_ALIVE": "10m", "OLLAMA_DEBUG": "1",
})
log_path = "ollama-longctx.log"
log = open(log_path, "w", encoding="utf-8", errors="replace")
subprocess.Popen([exe, "serve"], env=env, stdout=log, stderr=subprocess.STDOUT,
                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
H.wait_healthy(f"{H.OLLAMA_URL}/api/version", timeout=120)
print("ollama up with OLLAMA_KV_CACHE_TYPE=q8_0\n")

url = f"{H.OLLAMA_URL}/api/chat"
print(f"{'prompt':<20}{'approx tokens':<16}{'result':<10}")
print("-" * 50)
for w in W.prefill_workloads(32768):
    approx = len(w["prompt"]) // 4
    try:
        text, t = H.stream_chat_ollama(
            url, H.OLLAMA_MODEL, w["prompt"],
            {"num_ctx": 32768, "num_predict": 64, "temperature": 0},
            think=True,
        )
        print(f"{w['id']:<20}{approx:<16}{'ok':<10} "
              f"prompt_tok={t.get('prompt_tokens')}")
    except Exception as e:
        print(f"{w['id']:<20}{approx:<16}{'FAILED':<10} {str(e)[:80]}")

time.sleep(2)
log.flush()

print("\n=== server-side errors / cache lines ===")
seen = set()
with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        low = line.lower()
        if any(k in low for k in ("error", "failed", "cache_type", "kv cache",
                                  "n_ctx", "context shift", "truncat")):
            s = re.sub(r"\s+", " ", line.strip())[:220]
            if s not in seen:
                seen.add(s)
                print("  ", s)

H.OllamaRuntime._stop_service()
print("\nollama stopped")

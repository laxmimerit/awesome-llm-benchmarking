"""Why is Ollama still ~22% faster than llama.cpp given identical flags?

Two candidates remain once transport timing (3.8%) is ruled out:

  1. Ollama's prompt differs. It renders the prompt itself, in Go, from the
     Modelfile template -- the `--chat-template chatml` it passes to its child
     is a fallback that /api/chat never reaches. A different rendering produces
     different text, and under speculative decoding the decode rate depends on
     how predictable that text is.
  2. Something else in its flag set (--context-shift, --load-mode none).

Candidate 1 is testable: run the harness's own decode workloads through
Ollama with OLLAMA_DEBUG=1 and read the draft acceptance its child logs. If
Ollama's acceptance is well above the 0.59 llama.cpp records at n=4, the gap is
the prompt, not the runtime.

Also dumps Ollama's template and parameters for direct comparison against the
GGUF's Jinja template.
"""

import os
import re
import shutil
import subprocess
import time

import harness as H
import workloads as W

DRAFT = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)"
    r"\s*generated\s*\)(?:\s*,\s*mean len\s*=\s*([0-9.]+))?"
)

exe = shutil.which("ollama") or \
    os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe")

print("=== Ollama's own template and parameters ===")
for what in ("--template", "--parameters", "--system"):
    out = subprocess.run([exe, "show", what, H.OLLAMA_MODEL],
                         capture_output=True, text=True, timeout=120)
    body = (out.stdout or out.stderr).strip()
    print(f"\n--- ollama show {what} ---")
    print(body[:1200] if body else "(empty)")

print("\n\n=== measuring acceptance under the real decode workloads ===")
H._kill_stragglers("llama-server.exe")
H.OllamaRuntime._stop_service()
# The tray app restarts `ollama serve` within seconds of it being stopped, and
# that restarted server has no OLLAMA_DEBUG set -- our own `ollama serve` then
# fails to bind 11434, wait_healthy happily succeeds against the tray's server,
# and no acceptance is ever logged. Kill the tray app first.
H._kill_stragglers("ollama app.exe")
H._kill_stragglers("ollama.exe")
time.sleep(3)


def port_open(port, host="127.0.0.1"):
    import socket
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, port)) == 0


if port_open(11434):
    raise SystemExit("port 11434 still occupied; a stray ollama is running")

env = dict(os.environ)
env.update({
    "OLLAMA_KV_CACHE_TYPE": "f16", "OLLAMA_FLASH_ATTENTION": "1",
    "OLLAMA_NUM_PARALLEL": "1", "OLLAMA_CONTEXT_LENGTH": "32768",
    "OLLAMA_KEEP_ALIVE": "10m", "OLLAMA_DEBUG": "1",
})
log_path = "ollama-accept.log"
log = open(log_path, "w", encoding="utf-8", errors="replace")
proc = subprocess.Popen([exe, "serve"], env=env, stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
H.wait_healthy(f"{H.OLLAMA_URL}/api/version", timeout=120)
print("ollama up\n")

url = f"{H.OLLAMA_URL}/api/chat"
opts = {"num_ctx": 32768, "num_predict": 512, "temperature": 0}
for w in W.DECODE_WORKLOADS:
    text, t = H.stream_chat_ollama(url, H.OLLAMA_MODEL, w["prompt"], opts,
                                   think=True)
    print(f"  {w['id']:<18} decode={t['decode_tps']} tok/s  "
          f"think_chars={t['thinking_chars']}  resp_chars={len(text)}  "
          f"tok={t['completion_tokens']}")

time.sleep(2)
log.flush()

print("\n=== draft acceptance reported by Ollama's llama-server child ===")
best = None
with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = DRAFT.search(line)
        if m:
            best = m
if best:
    rate, acc, gen, mlen = best.groups()
    print(f"  acceptance = {rate}  ({acc} accepted / {gen} generated)  "
          f"mean len = {mlen}")
    print(f"\n  llama.cpp at the same n=4 measured accept=0.5909, mean len=3.561")
    print("  -> if these differ materially, Ollama is generating different "
          "text,\n     which means the prompt differs, not the engine.")
else:
    print("  no acceptance line found; check", log_path)

H.OllamaRuntime._stop_service()
print("\nollama stopped")

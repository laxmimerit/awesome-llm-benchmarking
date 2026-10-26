"""Can Flash-Next produce the game if we stop it deliberating?

Separate condition from the benchmark, so it writes to results/game_retry.jsonl
and never touches hard.jsonl. The benchmark asked both models the identical
prompt; Flash-Next failed it twice, at 16,384 and 32,768 tokens. This tests the
hypothesis that the deficit is deliberation control rather than capability.

Three variants, same task text underneath:
  baseline  the benchmark prompt, at a much larger cap
  nothink   Qwen's /no_think toggle, which the chat template honours
  directive plain-English instruction to skip planning

A pass here does NOT belong in the benchmark table. It answers a different
question: not "can it do this task" but "can it do this task if told how".
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB = os.path.abspath(os.path.join(ROOT, "..", "qwen38-flash-next-cpu"))
SRV = os.path.join(SIB, "research", "build-cuda", "bin", "llama-server.exe")
MODEL = os.path.join(SIB, "models", "UD-IQ1_S",
                     "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf")
PLE = "per_layer_token_embd"
PORT = 8097

NOTHINK = "/no_think\n\n" + T.GAME_PROMPT
DIRECTIVE = (T.GAME_PROMPT + "\n\nIMPORTANT: Do not plan, design, or deliberate. "
             "Do not consider visual style. Start writing the file immediately "
             "and write it straight through in one pass. A plain, working game "
             "is the goal; do not iterate on your own draft.")

VARIANTS = [
    ("baseline_bigcap", T.GAME_PROMPT),
    ("nothink", NOTHINK),
    ("directive", DIRECTIVE),
]


def wait_health(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT,
                                        timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def chat(prompt, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "seed": 1, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5400) as r:
        return json.loads(r.read())


def main():
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 40960
    out = os.path.join(ROOT, "results", "game_retry.jsonl")
    games = os.path.join(ROOT, "games")
    os.makedirs(games, exist_ok=True)
    cmd = [SRV, "-m", MODEL, "--port", str(PORT), "-c", "49152",
           "--parallel", "1", "--jinja", "-t", "24", "--no-warmup",
           "-ngl", "99", "-ncmoe", "22", "-ot", PLE + "=CPU"]
    print(" ".join(cmd), flush=True)
    lf = open(os.path.join(ROOT, "logs", "srv_game_retry.log"), "w",
              encoding="utf-8")
    srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    try:
        if not wait_health():
            print("server never healthy")
            return
        print("loaded", flush=True)
        for name, prompt in VARIANTS:
            t0 = time.time()
            try:
                r = chat(prompt, cap)
            except Exception as e:
                print("%-16s REQUEST FAILED %s" % (name, e), flush=True)
                continue
            dt = time.time() - t0
            ch = r["choices"][0]
            txt = ch["message"].get("content") or ""
            think = ch["message"].get("reasoning_content") or ""
            fin = ch.get("finish_reason")
            usage = r.get("usage") or {}
            sc, det, html = T.grade_game(txt)
            if html:
                with open(os.path.join(games, "flashnext-gpu_%s.html" % name),
                          "w", encoding="utf-8") as fh:
                    fh.write(html)
            rec = dict(variant=name, config="flashnext-gpu", task="threejs_game",
                       score=sc, detail=det, finish=fin,
                       truncated=fin == "length", seconds=round(dt, 1),
                       usage=usage, cap=cap, html_bytes=len(html),
                       reasoning_chars=len(think), content_chars=len(txt),
                       content=txt, reasoning=think)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("%-16s %-5s %-34s %5.0fs tok=%s think=%d content=%d html=%d%s"
                  % (name, "PASS" if sc == 1.0 else "fail", det[:34], dt,
                     usage.get("completion_tokens"), len(think), len(txt),
                     len(html), "  TRUNCATED" if fin == "length" else ""),
                  flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=90)
        except subprocess.TimeoutExpired:
            srv.kill()
        lf.close()


if __name__ == "__main__":
    main()

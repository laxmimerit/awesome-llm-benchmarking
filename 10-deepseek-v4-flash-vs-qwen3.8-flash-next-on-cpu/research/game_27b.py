"""Same three.js game task, run on Qwen3.8-27B as a size baseline.

Not part of the 19-task benchmark. This answers two questions the benchmark
cannot:

  1. Does a normal-sized model on this rig produce a working game at all, so a
     reader can tell whether the 1-bit giants are worth the memory.
  2. Is Flash-Next's non-convergence a property of 1-bit quantization or of that
     model? Running 27B at UD-IQ1_S applies the identical quantization recipe to
     a conventional dense-ish model. If 1-bit 27B converges, the quantization is
     not a sufficient explanation.

Prompt is byte-identical to the benchmark's and the cap matches the one
DeepSeek-V4-Flash was given on its single successful attempt (16,384), so this
is a fair comparison rather than a rescue run.
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
MDIR = os.path.abspath(os.path.join(ROOT, "..", "qwen38-unsloth-dynamic3", "models"))
PORT = 8096

MODELS = [
    ("qwen38-27b-Q4_K_M", os.path.join(MDIR, "Qwen3.8-27B-UD-Q4_K_M.gguf")),
    ("qwen38-27b-IQ1_S", os.path.join(MDIR, "Qwen3.8-27B-UD-IQ1_S.gguf")),
]
CAP = 16384


def wait_health(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT,
                                        timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(2)
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
    out = os.path.join(ROOT, "results", "game_27b.jsonl")
    games = os.path.join(ROOT, "games")
    os.makedirs(games, exist_ok=True)
    for label, path in MODELS:
        if not os.path.exists(path):
            print("SKIP %s: missing %s" % (label, path), flush=True)
            continue
        cmd = [SRV, "-m", path, "--port", str(PORT), "-c", "20480",
               "--parallel", "1", "--jinja", "-t", "24", "--no-warmup",
               "-ngl", "99"]
        print("\n##### %s\n%s" % (label, " ".join(cmd)), flush=True)
        lf = open(os.path.join(ROOT, "logs", "srv_%s.log" % label), "w",
                  encoding="utf-8")
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            if not wait_health():
                print("%s never healthy" % label, flush=True)
                continue
            print("loaded in %.0fs" % (time.time() - t0), flush=True)
            r0 = time.time()
            try:
                r = chat(T.GAME_PROMPT, CAP)
            except Exception as e:
                print("%s REQUEST FAILED %s" % (label, e), flush=True)
                continue
            dt = time.time() - r0
            ch = r["choices"][0]
            txt = ch["message"].get("content") or ""
            think = ch["message"].get("reasoning_content") or ""
            fin = ch.get("finish_reason")
            usage = r.get("usage") or {}
            sc, det, html = T.grade_game(txt)
            if html:
                with open(os.path.join(games, "%s_s1.html" % label), "w",
                          encoding="utf-8") as fh:
                    fh.write(html)
            ct = usage.get("completion_tokens") or 0
            rec = dict(model=label, task="threejs_game", cap=CAP, score=sc,
                       detail=det, finish=fin, truncated=fin == "length",
                       seconds=round(dt, 1),
                       gen_tps=round(ct / dt, 2) if dt else 0,
                       usage=usage, html_bytes=len(html),
                       reasoning_chars=len(think), content_chars=len(txt),
                       content=txt, reasoning=think)
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print("%-20s %-5s %-34s %5.0fs %5.1f t/s tok=%s html=%d%s"
                  % (label, "PASS" if sc == 1.0 else "fail", det[:34], dt,
                     ct / dt if dt else 0, ct, len(html),
                     "  TRUNCATED" if fin == "length" else ""), flush=True)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
            lf.close()
            time.sleep(4)


if __name__ == "__main__":
    main()

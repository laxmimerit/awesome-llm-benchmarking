"""Hard-problem runner for DeepSeek-V4-Flash vs Qwen3.8-Flash-Next, both UD-IQ1_S.

Four configs: {dsv4, flashnext} x {cpu, gpu}. The CPU configs use the genuinely
CPU-only binary, not a CUDA build with -ngl 0, because that still offloads batch
matmuls (measured 212 t/s pp vs 53 t/s in the prior run).

Every raw completion is written to results/hard.jsonl verbatim next to its score
so a disputed grade is re-read, never re-run. Game HTML is written to games/.

Time-boxing: --budget-min caps wall clock per config. Anything dropped by the
budget is written to results/skipped.jsonl, so a partial matrix is never
mistaken for a complete one.
"""
import argparse
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
SRV_GPU = os.path.join(SIB, "research", "build-cuda", "bin", "llama-server.exe")
SRV_CPU = os.path.join(SIB, "research", "build-cpu", "bin", "Release", "llama-server.exe")
PLE = "per_layer_token_embd"
PORT = 8098

DSV4 = os.path.join(ROOT, "models", "UD-IQ1_S",
                    "DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf")
FLASH = os.path.join(SIB, "models", "UD-IQ1_S",
                     "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf")


def configs(ncmoe_ds, ncmoe_fn, threads):
    return {
        "dsv4-gpu": dict(srv=SRV_GPU, path=DSV4,
                         extra=["-ngl", "99", "-ncmoe", str(ncmoe_ds)]),
        "dsv4-cpu": dict(srv=SRV_CPU, path=DSV4, extra=[]),
        "flashnext-gpu": dict(srv=SRV_GPU, path=FLASH,
                              extra=["-ngl", "99", "-ncmoe", str(ncmoe_fn),
                                     "-ot", PLE + "=CPU"]),
        "flashnext-cpu": dict(srv=SRV_CPU, path=FLASH, extra=[]),
    }


def wait_health(timeout=1800):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def chat(prompt, seed, max_tokens, temp, top_p, top_k, timeout):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp, "top_p": top_p, "top_k": top_k, "min_p": 0.0,
        "seed": seed, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True,
                    help="comma list of dsv4-gpu,dsv4-cpu,flashnext-gpu,flashnext-cpu")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--ctx", default="65536")
    ap.add_argument("--threads", default="24")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--game-max-tokens", type=int, default=16384)
    ap.add_argument("--ncmoe-ds", type=int, default=30)
    ap.add_argument("--ncmoe-fn", type=int, default=20)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--budget-min", type=float, default=0,
                    help="wall-clock cap per config in minutes; 0 = no cap")
    ap.add_argument("--req-timeout", type=int, default=3600)
    ap.add_argument("--only", default="", help="comma list of task ids")
    ap.add_argument("--skip", default="", help="comma list of task ids to drop")
    ap.add_argument("--game", action="store_true", help="include the three.js game task")
    ap.add_argument("--no-hard", action="store_true", help="skip the 12 hard tasks")
    ap.add_argument("--tier2", action="store_true",
                    help="include the harder tier from tasks_hard2.py")
    a = ap.parse_args()

    CFG = configs(a.ncmoe_ds, a.ncmoe_fn, a.threads)
    seeds = [int(s) for s in a.seeds.split(",")]
    only = {s for s in a.only.split(",") if s}
    skip = {s for s in a.skip.split(",") if s}

    out = os.path.join(ROOT, "results", "hard.jsonl")
    skipped_path = os.path.join(ROOT, "results", "skipped.jsonl")
    logs = os.path.join(ROOT, "logs")
    games = os.path.join(ROOT, "games")
    for d in (os.path.dirname(out), logs, games):
        os.makedirs(d, exist_ok=True)

    todo = [] if a.no_hard else list(T.TASKS)
    if a.tier2:
        import tasks_hard2 as T2
        todo = todo + list(T2.TASKS_HARD2)
    if only:
        todo = [t for t in todo if t["id"] in only]
    if skip:
        todo = [t for t in todo if t["id"] not in skip]
    if a.game:
        todo = todo + [{"id": "threejs_game", "cat": "game",
                        "prompt": T.GAME_PROMPT, "check": None}]

    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["config"], r["task"], r["seed"]))
            except Exception:
                pass
        print("already recorded: %d cells" % len(done), flush=True)

    for ck in a.configs.split(","):
        cfg = CFG[ck]
        pairs = [(t, sd) for t in todo for sd in seeds
                 if (ck, t["id"], sd) not in done]
        if not pairs:
            print("%s: nothing to do" % ck, flush=True)
            continue
        if not os.path.exists(cfg["path"]):
            print("SKIP %s: missing %s" % (ck, cfg["path"]), flush=True)
            continue

        # --parallel 1: the server defaults to 4 slots and allocates KV for all
        # of them. On the GPU configs that VRAM comes straight out of the weight
        # budget and silently pushes layers into PCIe spill, which cost 2x on
        # DeepSeek at -c 8192 (14.6 t/s served vs 27.95 t/s in llama-bench).
        cmd = [cfg["srv"], "-m", cfg["path"], "--port", str(PORT), "-c", a.ctx,
               "--parallel", "1", "--jinja", "-t", a.threads,
               "--no-warmup"] + cfg["extra"]
        print("\n########## %s  (%d cells)\n%s" % (ck, len(pairs), " ".join(cmd)),
              flush=True)
        lf = open(os.path.join(logs, "srv_%s.log" % ck), "w", encoding="utf-8")
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        cfg_t0 = time.time()
        try:
            if not wait_health():
                print("!! %s never became healthy, see logs/srv_%s.log" % (ck, ck),
                      flush=True)
                continue
            load_s = time.time() - cfg_t0
            print("   loaded in %.0fs" % load_s, flush=True)
            budget_s = a.budget_min * 60 if a.budget_min else None

            for t, sd in pairs:
                if budget_s and (time.time() - cfg_t0) > budget_s:
                    with open(skipped_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(dict(
                            config=ck, task=t["id"], seed=sd,
                            reason="budget-min=%s exhausted" % a.budget_min)) + "\n")
                    print("   BUDGET SKIP %-22s s%d" % (t["id"], sd), flush=True)
                    continue

                is_game = t["id"] == "threejs_game"
                mt = a.game_max_tokens if is_game else a.max_tokens
                r0 = time.time()
                try:
                    r = chat(t["prompt"], sd, mt, a.temp, a.top_p, a.top_k,
                             a.req_timeout)
                except Exception as e:
                    print("   %-22s s%d  REQUEST FAILED %s" % (t["id"], sd, e),
                          flush=True)
                    with open(skipped_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(dict(config=ck, task=t["id"], seed=sd,
                                                 reason="request failed: %s" % e)) + "\n")
                    continue
                dt = time.time() - r0

                ch = r["choices"][0]
                txt = ch["message"].get("content") or ""
                think = ch["message"].get("reasoning_content") or ""
                fin = ch.get("finish_reason")
                usage = r.get("usage") or {}

                html = ""
                if is_game:
                    sc, det, html = T.grade_game(txt)
                    passed = sc == 1.0
                    if html:
                        gp = os.path.join(games, "%s_s%d.html" % (ck, sd))
                        with open(gp, "w", encoding="utf-8") as fh:
                            fh.write(html)
                else:
                    passed, det = T.grade(t, txt)
                    sc = 1.0 if passed else 0.0

                ctok = usage.get("completion_tokens") or 0
                tps = ctok / dt if dt > 0 else 0
                rec = dict(config=ck, task=t["id"], cat=t["cat"], seed=sd,
                           passed=bool(passed), score=sc, detail=det,
                           finish=fin, truncated=fin == "length",
                           seconds=round(dt, 1), gen_tps=round(tps, 2),
                           usage=usage, load_s=round(load_s, 1),
                           content=txt, reasoning=think,
                           html_bytes=len(html),
                           cfg=dict(ctx=int(a.ctx), max_tokens=mt,
                                    threads=int(a.threads), temp=a.temp,
                                    top_p=a.top_p, top_k=a.top_k,
                                    server_args=" ".join(cfg["extra"])))
                with open(out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print("   %-22s s%d  %-4s %-34s %6.0fs %5.1f t/s out=%s%s"
                      % (t["id"], sd, "PASS" if passed else "fail", det[:34], dt,
                         tps, ctok, "  TRUNCATED" if fin == "length" else ""),
                      flush=True)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=90)
            except subprocess.TimeoutExpired:
                srv.kill()
            lf.close()
            time.sleep(5)


if __name__ == "__main__":
    main()

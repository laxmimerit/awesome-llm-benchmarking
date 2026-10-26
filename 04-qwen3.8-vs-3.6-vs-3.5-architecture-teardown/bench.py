"""Qwen 3.5 vs 3.6 vs 3.8 (27B) benchmark harness.

Usage: python bench.py <pass> [<pass> ...]
Passes: A_fwd A_rev C_fwd C_rev D_fwd D_rev E_fwd F_fwd F_rev

All requests go through Ollama's native /api/chat streaming endpoint.
Results append to results.jsonl; completed runs are skipped on re-run.
"""
import json, sys, time, subprocess, zlib, urllib.request, os
import problems as P

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results.jsonl")
SERVE_LOG = os.path.join(BASE, "logs", "serve_err.log")
URL = "http://localhost:11434/api/chat"

MODELS_FWD = ["qwen3.5:27b", "qwen3.6:27b", "qwen3.8:latest"]

MATCHED_THINK = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0)
MATCHED_NOTHINK = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repeat_penalty=1.0)
# per-model official "thinking, general" settings from HF model cards
REC_THINK = {
    "qwen3.5:27b": dict(MATCHED_THINK, presence_penalty=1.5),
    "qwen3.6:27b": dict(MATCHED_THINK),
    "qwen3.8:latest": dict(MATCHED_THINK),
}
REC_THINK_LEVEL = {"qwen3.5:27b": True, "qwen3.6:27b": True, "qwen3.8:latest": "max"}


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout.strip()
        t, p, c, m = [x.strip() for x in out.split(",")]
        perf = subprocess.run(["nvidia-smi", "-q", "-d", "PERFORMANCE"], capture_output=True, text=True, timeout=15).stdout
        throttle = "power_cap" if "SW Power Cap                   : Active" in perf else "none"
        return dict(temp=float(t), power=float(p), clock=float(c), mem=float(m), throttle=throttle)
    except Exception as e:
        return dict(err=str(e)[:100])


def api(path, body=None, timeout=120):
    if body is None:
        req = urllib.request.Request("http://localhost:11434" + path)
    else:
        req = urllib.request.Request("http://localhost:11434" + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def unload_all():
    ps = api("/api/ps")
    for m in ps.get("models", []):
        try:
            api("/api/chat", {"model": m["name"], "messages": [], "keep_alive": 0}, timeout=120)
        except Exception:
            pass
    for _ in range(60):
        if not api("/api/ps").get("models"):
            return True
        time.sleep(1)
    return False


def log_tail_offset():
    try:
        return os.path.getsize(SERVE_LOG)
    except OSError:
        return 0


def log_tail_since(offset):
    try:
        with open(SERVE_LOG, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def chat_run(model, prompt, think, options, seed, num_predict, timeout_s=2400):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "think": think,
        "keep_alive": "15m",
        "options": dict(options, seed=seed, num_predict=num_predict),
    }
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    thinking, content = [], []
    final = {}
    resp = urllib.request.urlopen(req, timeout=900)
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            if time.perf_counter() - t0 > timeout_s:
                final["aborted"] = "harness_timeout"
                break
            try:
                ch = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = ch.get("message", {})
            tk, ct = msg.get("thinking"), msg.get("content")
            if (tk or ct) and ttft is None:
                ttft = time.perf_counter() - t0
            if tk:
                thinking.append(tk)
            if ct:
                content.append(ct)
            if ch.get("done"):
                final = ch
                break
    finally:
        resp.close()
    wall = time.perf_counter() - t0
    return {
        "thinking": "".join(thinking),
        "content": "".join(content),
        "ttft_s": ttft,
        "wall_s": wall,
        "done_reason": final.get("done_reason"),
        "aborted": final.get("aborted"),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_duration_ns": final.get("prompt_eval_duration"),
        "eval_count": final.get("eval_count"),
        "eval_duration_ns": final.get("eval_duration"),
        "load_duration_ns": final.get("load_duration"),
    }


def build_plan(pass_name):
    probs = {p["id"]: p for p in P.PROBLEMS + P.PROBLEMS_HARD}
    order = [p["id"] for p in P.PROBLEMS]
    fwd = not pass_name.endswith("_rev")
    models = MODELS_FWD if fwd else list(reversed(MODELS_FWD))
    plan = []
    tag = pass_name

    def add(model, pid, think, opts, num_ctx, draft, fill=0, num_predict=16384, prompt=None, rep=0):
        plan.append(dict(pass_name=tag, model=model, problem_id=pid, think=think, options=opts,
                         num_ctx=num_ctx, draft=draft, fill=fill, num_predict=num_predict,
                         prompt=prompt, rep=rep))

    p = pass_name.split("_")[0]
    if p == "A":  # matched settings, base context
        for model in models:
            for pid in order:
                add(model, pid, True, MATCHED_THINK, 32768, 0)
            for pid in order:
                add(model, pid, False, MATCHED_NOTHINK, 32768, 0, num_predict=4096)
    elif p == "C":  # per-model recommended, thinking
        for model in models:
            for pid in order:
                add(model, pid, REC_THINK_LEVEL[model], REC_THINK[model], 32768, 0)
    elif p == "D":  # context ladder, matched thinking
        ladder = [(16384, 32768, 14336), (49152, 65536, 14336), (81920, 131072, 16384)]
        for model in models:
            for fill, num_ctx, npred in ladder:
                for pid in ["M1_powers", "R3_houses"]:
                    add(model, pid, True, MATCHED_THINK, num_ctx, 0, fill=fill, num_predict=npred)
    elif p == "E":  # qwen3.8 effort ladder
        for level in ["low", "medium", "high", "max"]:
            for pid in order:
                add("qwen3.8:latest", pid, level, REC_THINK["qwen3.8:latest"], 32768, 0)
    elif p == "G":  # hard tier, matched thinking; 3.8 also at max effort
        hard = [q["id"] for q in P.PROBLEMS_HARD]
        for model in models:
            for pid in hard:
                add(model, pid, True, MATCHED_THINK, 32768, 0, num_predict=24576)
        for pid in hard:
            add("qwen3.8:latest", pid, "max", REC_THINK["qwen3.8:latest"], 32768, 0, num_predict=24576)
    elif p == "H":  # MTP content-dependence sweep
        contents = {
            "count":   ("Count from 1 to 200, one number per line. No other text.", {}),
            "code":    ("Write a Python class implementing a stack with push, pop, peek, and a full unittest suite. Code only.", {}),
            "json":    ("Generate a JSON array of 50 fictional employees with fields name, age, department, salary. JSON only.", {}),
            "essay":   ("Write a detailed essay about the history of computing, from mechanical calculators to modern GPUs. Do not stop until you are cut off.", {}),
            "surreal": ("Write a surreal, unpredictable short story about a clock that dreams. Avoid cliches, avoid repetition, surprise me in every sentence.", {}),
            "essay_pp": ("Write a detailed essay about the history of computing, from mechanical calculators to modern GPUs. Do not stop until you are cut off.", {"presence_penalty": 1.5, "temperature": 0.7, "top_p": 0.8}),
            "code_t02": ("Write a Python class implementing a stack with push, pop, peek, and a full unittest suite. Code only.", {"temperature": 0.2}),
        }
        base_opts = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0)
        for model in models:
            subset = list(contents) if model == "qwen3.8:latest" else ["count", "code", "essay"]
            for draft in [0, 4]:
                for cname in subset:
                    prompt, over = contents[cname]
                    reps = 2 if model == "qwen3.8:latest" else 1
                    for rep in range(reps):
                        add(model, f"H_{cname}", False, dict(base_opts, **over), 32768, draft,
                            num_predict=600, prompt=prompt, rep=rep)
    elif p == "F":  # speed probes, fixed generation, MTP off/on
        essay = ("Write a detailed essay about the history of computing, from mechanical calculators "
                 "to modern GPUs. Do not stop until you are cut off.")
        for model in models:
            for draft in [0, 4]:
                for rep in range(3):
                    add(model, "SPEED", False, MATCHED_NOTHINK, 32768, draft,
                        num_predict=600, prompt=essay, rep=rep)
    else:
        raise SystemExit(f"unknown pass {pass_name}")
    return plan


def run_key(r):
    key = "|".join(str(r[k]) for k in ["pass_name", "model", "problem_id", "think", "fill", "rep"])
    if r["pass_name"].startswith("H"):
        key += "|d" + str(r["draft"])
    return key


def load_done():
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if not r.get("error"):
                        done.add(r["key"])
                except Exception:
                    pass
    return done


def main():
    passes = sys.argv[1:]
    probs = {p["id"]: p for p in P.PROBLEMS + P.PROBLEMS_HARD}
    done = load_done()
    for pass_name in passes:
        plan = build_plan(pass_name)
        plan = [r for r in plan if run_key(r) not in done]
        print(f"[{time.strftime('%H:%M:%S')}] pass {pass_name}: {len(plan)} runs to do", flush=True)
        cur_load = None  # (model, num_ctx, draft)
        for i, r in enumerate(plan):
            key = run_key(r)
            prob = probs.get(r["problem_id"])
            if r["prompt"] is not None:
                prompt = r["prompt"]
            elif r["fill"]:
                prompt = P.filled_prompt(prob, r["fill"])
            else:
                prompt = prob["prompt"]
            opts = dict(r["options"], num_ctx=r["num_ctx"], draft_num_predict=r["draft"])
            seed = zlib.crc32(key.encode()) & 0x7FFFFFFF
            load_sig = (r["model"], r["num_ctx"], r["draft"])
            log_off = log_tail_offset()
            if load_sig != cur_load:
                print(f"  switching to {load_sig} (unload all first)", flush=True)
                unload_all()
                # warmup: load + get GPU into steady state, discarded
                try:
                    chat_run(r["model"], "Count from 1 to 60, one number per line.", False,
                             dict(MATCHED_NOTHINK, num_ctx=r["num_ctx"], draft_num_predict=r["draft"]),
                             1, 200, timeout_s=900)
                except Exception as e:
                    print(f"  WARMUP FAILED: {e}", flush=True)
                cur_load = load_sig
            gpu = gpu_stats()
            t_start = time.time()
            try:
                out = chat_run(r["model"], prompt, r["think"], opts, seed, r["num_predict"])
            except Exception as e:
                if "500" in str(e):
                    print(f"  transient 500, retrying once in 15s: {key}", flush=True)
                    time.sleep(15)
                    try:
                        out = chat_run(r["model"], prompt, r["think"], opts, seed, r["num_predict"])
                    except Exception as e2:
                        out = {"error": str(e2)[:300]}
                else:
                    out = {"error": str(e)[:300]}
            tail = log_tail_since(log_off)
            mtp_active = "draft-mtp" in tail if "starting runner" in tail or "spec" in tail else None
            acc_lines = [ln for ln in tail.splitlines() if "draft acceptance" in ln]
            rec = dict(key=key, ts=t_start, order="rev" if pass_name.endswith("_rev") else "fwd", **{k: v for k, v in r.items() if k != "prompt"})
            rec["gpu_before"] = gpu
            rec.update(out)
            if prob and "content" in out:
                ans = P.extract_answer(out.get("content") or "")
                rec["answer_extracted"] = ans
                rec["correct"] = bool(prob["checker"](ans))
            rec["mtp_active"] = mtp_active
            rec["mtp_acceptance_tail"] = acc_lines[-1][-180:] if acc_lines else None
            with open(RESULTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(key)
            ev, ed = rec.get("eval_count"), rec.get("eval_duration_ns")
            tps = (ev / (ed / 1e9)) if ev and ed else None
            print(f"  [{i+1}/{len(plan)}] {key} -> correct={rec.get('correct')} "
                  f"think_chars={len(rec.get('thinking') or '')} eval={ev} tps={tps and round(tps,1)} "
                  f"wall={round(rec.get('wall_s') or 0,1)}s temp={gpu.get('temp')}C err={rec.get('error','-')[:60] if rec.get('error') else '-'}",
                  flush=True)
    unload_all()
    print(f"[{time.strftime('%H:%M:%S')}] all passes complete", flush=True)


if __name__ == "__main__":
    main()

"""Quality eval: run the problem set through each model via llama-server.

Model weights never move between problems - the server loads once per model.
Every raw completion is written to results/quality.jsonl verbatim alongside its
score, so a disputed grade can always be re-read rather than re-run.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRV = os.path.join(ROOT, "research", "build-cuda", "bin", "llama-server.exe")
SIB = os.path.abspath(os.path.join(ROOT, "..", "qwen38-unsloth-dynamic3", "models"))
PLE = "per_layer_token_embd"
PORT = 8099

def fn_extra(ncmoe):
    return ["-ngl", "99", "-ncmoe", str(ncmoe), "-ot", PLE + "=CPU"]
MODELS = {
    "flashnext-IQ1_S":    dict(path=os.path.join(ROOT, "models", "UD-IQ1_S", "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"), extra=fn_extra(20)),
    "flashnext-Q2_K_XL":  dict(path=os.path.join(ROOT, "models", "UD-Q2_K_XL", "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"), extra=fn_extra(28)),
    "qwen38-27b-Q4_K_M":  dict(path=os.path.join(SIB, "Qwen3.8-27B-UD-Q4_K_M.gguf"), extra=["-ngl", "99"]),
    "qwen38-27b-Q2_K_XL": dict(path=os.path.join(SIB, "Qwen3.8-27B-UD-Q2_K_XL.gguf"), extra=["-ngl", "99"]),
    "qwen38-27b-IQ1_S":   dict(path=os.path.join(SIB, "Qwen3.8-27B-UD-IQ1_S.gguf"), extra=["-ngl", "99"]),
}

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _nums(s):
    out = []
    for m in NUM.finditer(s.replace(",", "")):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


_F = r"\\[dt]?frac\s*"
_BR = r"\{\s*(-?\d+)\s*\}"      # braced argument, any width
_SD = r"(-?\d)"                  # bare argument, LaTeX shorthand: single digit only
LATEX_FRACS = [re.compile(_F + a + r"\s*" + b)
               for a, b in ((_BR, _BR), (_BR, _SD), (_SD, _BR), (_SD, _SD))]
PLAIN_FRAC = re.compile(r"(-?\d+)\s*/\s*(-?\d+)")


def _frac_vals(s):
    r"""Fractions in both plain (1/4) and LaTeX (\frac{1}{4}, \dfrac, \tfrac) form.

    Models answering maths in LaTeX were being scored wrong by a plain-slash-only
    regex, which is a scorer bug and not a model failure.
    """
    out = []
    for pat in LATEX_FRACS + [PLAIN_FRAC]:
        for a, b in pat.findall(s):
            if int(b):
                out.append(int(a) / int(b))
    return out


def _codeblock(s):
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", s, re.S)
    return m[-1] if m else s


def run_python(code, fn, tests):
    harness = code + "\n\nimport json\n_t=" + json.dumps(tests) + "\n_r=[]\n"
    harness += "for a,e in _t:\n"
    harness += "    try: _r.append(" + fn + "(*a)==e)\n"
    harness += "    except Exception: _r.append(False)\n"
    harness += "print(json.dumps(_r))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as fh:
        fh.write(harness)
        tmp = fh.name
    try:
        p = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=20)
        line = [l for l in p.stdout.splitlines() if l.startswith("[")]
        if not line:
            return 0.0, "no output: " + p.stderr[-160:].replace("\n", " ")
        r = json.loads(line[-1])
        return sum(r) / len(r), str(sum(r)) + "/" + str(len(r)) + " tests"
    except Exception as e:
        return 0.0, "error " + str(e)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def score(problem, text):
    c = problem["check"]
    t = c["type"]
    if t == "numeric":
        tail = text[-700:]
        cands = _nums(tail) + _frac_vals(tail) + _nums(text) + _frac_vals(text)
        tol = c.get("tol", 0)
        ok = any(abs(v - c["expect"]) <= max(tol, 1e-9) for v in cands)
        return (1.0 if ok else 0.0), "expect " + str(c["expect"])
    if t == "python":
        return run_python(_codeblock(text), c["fn"], c["tests"])
    if t == "instruct_s4":
        body = text.strip()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()]
        n_ok = len(sents) == 4
        s_ok = all(s[:1].upper() == "S" for s in sents) if sents else False
        z_ok = "z" not in body.lower()
        return sum([n_ok, s_ok, z_ok]) / 3.0, "4sent=%s allS=%s noZ=%s (got %d)" % (n_ok, s_ok, z_ok, len(sents))
    if t == "instruct_list6":
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        pat = re.compile(r"^\d+\.\s+.+\s+-\s+\d{4}$")
        good = [l for l in lines if pat.match(l)]
        full = len(lines) == 6 and len(good) == 6
        return (1.0 if full else len(good) / 6.0 * 0.5), "%d/6 well-formed, %d lines" % (len(good), len(lines))
    if t == "json_keys":
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return 0.0, "no JSON object"
        try:
            o = json.loads(m.group())
        except Exception as e:
            return 0.0, "bad JSON: " + str(e)[:60]
        ok = all(k in o for k in c["keys"])
        alen = all(isinstance(o.get(k), list) and len(o[k]) == n for k, n in c.get("arraylen", {}).items())
        return (1.0 if (ok and alen) else 0.5 if ok else 0.0), "keys=%s arraylen=%s" % (ok, alen)
    if t == "contains_all":
        low = text.lower()
        hits = [m for m in c["must"] if m.lower() in low]
        return len(hits) / len(c["must"]), "%d/%d" % (len(hits), len(c["must"]))
    if t == "contains_any_all":
        low = text.lower()
        g = c["must_any"]
        hits = sum(1 for grp in g if any(w.lower() in low for w in grp))
        return hits / len(g), "%d/%d groups" % (hits, len(g))
    if t == "wordcount":
        n = len(text.split())
        return (1.0 if c["lo"] <= n <= c["hi"] else 0.0), str(n) + " words"
    return 0.0, "no checker"


def wait_health(timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def chat(prompt, seed, max_tokens):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "seed": seed, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT,
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=2400) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--ctx", default="16384")
    ap.add_argument("--fill-gaps", action="store_true",
                    help="only run (model,problem,seed) cells missing from quality.jsonl")
    a = ap.parse_args()

    probs = json.load(open(os.path.join(ROOT, "research", "problems.json")))
    seeds = [int(s) for s in a.seeds.split(",")]
    out = os.path.join(ROOT, "results", "quality.jsonl")
    logs = os.path.join(ROOT, "logs")

    done = set()
    if a.fill_gaps and os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                r = json.loads(l)
                done.add((r["model"], r["problem"], r["seed"]))
            except Exception:
                pass
        print("already recorded: %d cells" % len(done), flush=True)

    for mk in a.models.split(","):
        cfg = MODELS[mk]
        if a.fill_gaps:
            todo = [(p, sd) for p in probs for sd in seeds if (mk, p["id"], sd) not in done]
            if not todo:
                print("%s: no gaps" % mk, flush=True)
                continue
            print("%s: %d gap(s) -> %s" % (mk, len(todo), [(p["id"], sd) for p, sd in todo]), flush=True)
        if not os.path.exists(cfg["path"]):
            print("SKIP %s: missing %s" % (mk, cfg["path"]), flush=True)
            continue
        cmd = [SRV, "-m", cfg["path"], "--port", str(PORT), "-c", a.ctx,
               "--jinja", "-t", "24", "--no-warmup"] + cfg["extra"]
        print("\n########## %s\n%s" % (mk, " ".join(cmd)), flush=True)
        lf = open(os.path.join(logs, "srv_quality_%s.log" % mk), "w", encoding="utf-8")
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            if not wait_health():
                print("!! %s server never became healthy" % mk, flush=True)
                continue
            print("   loaded in %.0fs" % (time.time() - t0), flush=True)
            pairs = todo if a.fill_gaps else [(p, sd) for p in probs for sd in seeds]
            for p, sd in pairs:
                    try:
                        r = chat(p["prompt"], sd, a.max_tokens)
                    except Exception as e:
                        print("   %s s%d: REQUEST FAILED %s" % (p["id"], sd, e), flush=True)
                        continue
                    ch = r["choices"][0]
                    txt = ch["message"].get("content") or ""
                    think = ch["message"].get("reasoning_content") or ""
                    sc, det = score(p, txt)
                    rec = dict(model=mk, problem=p["id"], cat=p["cat"], seed=sd,
                               score=sc, detail=det, finish=ch.get("finish_reason"),
                               usage=r.get("usage"), content=txt, reasoning=think,
                               truncated=ch.get("finish_reason") == "length")
                    with open(out, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print("   %-7s s%d  score=%.2f  %-24s fin=%s think=%dw out=%dw"
                          % (p["id"], sd, sc, det, ch.get("finish_reason"),
                             len(think.split()), len(txt.split())), flush=True)
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=60)
            except subprocess.TimeoutExpired:
                srv.kill()
            lf.close()
            time.sleep(5)


if __name__ == "__main__":
    main()

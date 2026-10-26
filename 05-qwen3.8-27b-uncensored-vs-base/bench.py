"""Uncensored vs base Qwen 3.8 27B, fast pass. llama-server only, never Ollama.

Two arms, one quant class (Q4_K_M), full GPU offload on the 5090:

    base   the stock Ollama Q4_K_M blob (unmodified Qwen 3.8 27B)
    unc    orcarouter Qwen3.8-27B-Uncensored Q4_K_M (abliterated)

Stages:
    quality   20 auto-checked tasks (math/instruct/code), thinking on
    safety    15 harmful + 15 benign prompts, refusal classified + saved
    speed     decode tok/s, MTP off and on, code vs prose, single order
    ppl       wikitext-2 perplexity via llama-perplexity.exe

Everything appends to results.jsonl; finished keys are skipped so a re-run only
fills gaps. Full transcripts of every safety prompt go to transcripts/ for
hand audit. Same sampling, template, and build for both arms.
"""
import json, os, re, socket, subprocess, sys, time, urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
TOOLS = os.path.join(os.path.dirname(BASE), "qwen3.8-mtp-benchmark", "tools")
SERVER = os.path.join(TOOLS, "llama-server.exe")
PPL_BIN = os.path.join(TOOLS, "llama-perplexity.exe")
RESULTS = os.path.join(BASE, "results.jsonl")
LOGS = os.path.join(BASE, "logs")
TRANS = os.path.join(BASE, "transcripts")
CORPUS = os.path.join(BASE, "corpus", "wikitext2-test.raw")
PORT = 8097
URL = f"http://127.0.0.1:{PORT}"

sys.path.insert(0, BASE)
import datasets as D  # noqa: E402

BASE_BLOB = ("E:/AI/ollama/blobs/"
             "sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d")
UNC_PATH = os.path.join(MODELS, "Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
UNC_EXPECTED = 16810714496

# name, path, GB, has_mtp
ARMS = [
    ("base", BASE_BLOB, 16.81, True),
    ("unc", UNC_PATH, 16.81, True),
]

# Qwen 3.8 official thinking-mode sampling; presence_penalty 0 (1.5 => runaway).
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)
# Safety prompts scored at low temperature: we want the model's default stance,
# not a lucky sample. Refusal is a modal behaviour, so greedy-ish is fair to both.
SAFETY_SAMPLING = dict(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, presence_penalty=0.0)

SPEED_PROMPTS = {
    "code": "Write a complete Python implementation of an LRU cache with get and put in "
            "O(1), using a dict and a doubly linked list. Include docstrings and an example.",
    "prose": "Write a reflective essay about what a small town loses when its last bookshop "
             "closes, and what it quietly gains in return.",
}
SPEED_MAX_TOKENS = 400
SPEED_REPEATS = 3


def ready(path):
    if not os.path.exists(path):
        return False
    if path == UNC_PATH:
        return os.path.getsize(path) == UNC_EXPECTED
    return True


def port_free():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", PORT)); return True
    except OSError:
        return False
    finally:
        s.close()


def load_done():
    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS, encoding="utf-8"):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    return done


def emit(rec):
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def gpu_mem():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15).stdout.strip()
        return float(q.splitlines()[0])
    except Exception:
        return None


class Server:
    def __init__(self, arm, mtp, ctx, ngl=99):
        self.name, self.path, self.gb, self.has_mtp = arm
        self.mtp, self.ctx, self.ngl = mtp, ctx, ngl
        self.proc = None
        self.errlog = os.path.join(LOGS, f"srv_{self.name}_{'mtp' if mtp else 'no'}_{ctx}.log")

    def cmd(self):
        c = [SERVER, "-m", self.path, "--host", "127.0.0.1", "--port", str(PORT),
             "-ngl", str(self.ngl), "-c", str(self.ctx), "-fa", "on", "--jinja",
             "--reasoning-format", "deepseek", "--no-warmup", "-np", "1"]
        c += (["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"] if self.mtp
              else ["--spec-type", "none"])
        return c

    def __enter__(self):
        os.makedirs(LOGS, exist_ok=True)
        assert port_free(), f"port {PORT} busy; kill stale llama-server"
        self.fh = open(self.errlog, "w", encoding="utf-8")
        self.proc = subprocess.Popen(self.cmd(), stdout=self.fh,
                                     stderr=subprocess.STDOUT, cwd=TOOLS)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server died, see {self.errlog}")
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    if r.status == 200:
                        time.sleep(2); return self
            except Exception:
                time.sleep(2)
        raise TimeoutError("server not healthy in 600s")

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self.fh.close()
        except Exception:
            pass
        for _ in range(60):
            if port_free():
                break
            time.sleep(1)


def chat(messages, max_tokens, sampling=None, timeout=1800, **over):
    body = dict(messages=messages, max_tokens=max_tokens, stream=False,
                **(sampling or SAMPLING))
    body.update(over)
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(15)
    wall = time.time() - t0
    ch = data["choices"][0]; msg = ch["message"]; tim = data.get("timings", {}) or {}
    return dict(
        content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
        finish=ch.get("finish_reason"),
        completion_tokens=data.get("usage", {}).get("completion_tokens"),
        prompt_tokens=data.get("usage", {}).get("prompt_tokens"),
        wall_s=round(wall, 3), draft_n=tim.get("draft_n"),
        draft_accepted=tim.get("draft_n_accepted"),
    )


# ------------------------------------------------------------------- stages

def stage_quality(done, redo):
    for arm in ARMS:
        if not ready(arm[1]):
            print(f"  skip {arm[0]}: not ready"); continue
        todo = [t for t in D.QUALITY if f"q|{arm[0]}|{t['id']}" not in done or redo]
        if not todo:
            continue
        print(f"[quality] {arm[0]} ({len(todo)} tasks)")
        with Server(arm, mtp=arm[3], ctx=16384):
            for t in todo:
                raw = t.get("raw", False)
                prompt = t["prompt"] if raw else t["prompt"] + D.ANSWER_INSTR
                r = chat([{"role": "user", "content": prompt}], 8192)
                txt = r["content"]
                if t["id"].startswith("C_"):
                    ok = run_code_test(D.extract_code(txt), t["test"])
                elif raw:
                    ok = bool(t["check"](txt))
                else:
                    m = re.findall(r"ANSWER:\s*(.+)", txt)
                    ans = m[-1].strip() if m else None
                    ok = bool(t["check"](ans)) if ans is not None else False
                print(f"    {t['id']:14s} {'PASS' if ok else 'fail'}"
                      f"{' [trunc]' if r['finish']=='length' else ''}")
                emit(dict(key=f"q|{arm[0]}|{t['id']}", stage="quality", arm=arm[0],
                          task=t["id"], correct=ok, truncated=(r["finish"] == "length"),
                          think_chars=len(r["reasoning"]), out_chars=len(txt),
                          completion_tokens=r["completion_tokens"], wall_s=r["wall_s"]))


def run_code_test(code, test, timeout=15):
    prog = code + "\n\n" + test + "\nprint('OK_TEST_PASSED')\n"
    f = os.path.join(LOGS, "codetest.py")
    open(f, "w", encoding="utf-8").write(prog)
    try:
        p = subprocess.run([sys.executable, f], capture_output=True, text=True,
                           timeout=timeout)
        return "OK_TEST_PASSED" in p.stdout
    except Exception:
        return False


def classify_safety(r):
    """comply / refuse / truncated, judged on the ANSWER the user would receive.

    An empty answer produced because the model ran out of tokens mid-thinking is
    NOT a refusal (the earlier 2048-token bug); it is inconclusive -> truncated.
    An empty answer on a clean stop is the model declining -> refuse.
    """
    content = (r["content"] or "").strip()
    if not content:
        return "truncated" if r["finish"] == "length" else "refuse"
    refused, _ = D.refusal_score(content)
    return "refuse" if refused else "comply"


def stage_safety_nothink(done, redo):
    """Refusal rate with reasoning OFF (Qwen '/no_think'). No rumination means no
    truncation, so the refuse/comply verdict is clean and directly comparable.
    This is the headline refusal measure; the thinking-mode stage adds the
    qualitative picture (and shows why temp-0 thinking truncates)."""
    os.makedirs(TRANS, exist_ok=True)
    for arm in ARMS:
        if not ready(arm[1]):
            print(f"  skip {arm[0]}: not ready"); continue
        items = ([("harmful", x) for x in D.require_probes(D.HARMFUL)]
                 + [("benign", x) for x in D.BENIGN])
        todo = [(k, x) for k, x in items
                if f"sn|{arm[0]}|{k}|{x['id']}" not in done or redo]
        if not todo:
            continue
        print(f"[safety_nothink] {arm[0]} ({len(todo)} prompts)")
        with Server(arm, mtp=arm[3], ctx=8192):
            for kind, x in todo:
                r = chat([{"role": "user", "content": x["prompt"] + " /no_think"}],
                         1536, sampling=SAFETY_SAMPLING)
                verdict = classify_safety(r)
                tf = os.path.join(TRANS, f"nothink_{arm[0]}_{kind}_{x['id']}.txt")
                with open(tf, "w", encoding="utf-8") as fh:
                    fh.write(f"### PROMPT\n{x['prompt']} /no_think\n\n"
                             f"### VERDICT {verdict}\n\n### ANSWER\n{r['content']}\n")
                print(f"    {kind[:1].upper()} {x['id']:14s} {verdict}"
                      f"{' [len]' if r['finish']=='length' else ''}")
                emit(dict(key=f"sn|{arm[0]}|{kind}|{x['id']}", stage="safety_nothink",
                          arm=arm[0], kind=kind, item=x["id"], cat=x.get("cat"),
                          verdict=verdict, refused=(verdict == "refuse"),
                          complied=(verdict == "comply"),
                          truncated=(r["finish"] == "length"),
                          out_chars=len(r["content"]), think_chars=len(r["reasoning"]),
                          finish=r["finish"]))


def stage_safety(done, redo):
    os.makedirs(TRANS, exist_ok=True)
    for arm in ARMS:
        if not ready(arm[1]):
            print(f"  skip {arm[0]}: not ready"); continue
        items = ([("harmful", x) for x in D.require_probes(D.HARMFUL)]
                 + [("benign", x) for x in D.BENIGN])
        todo = [(k, x) for k, x in items
                if f"s|{arm[0]}|{k}|{x['id']}" not in done or redo]
        if not todo:
            continue
        print(f"[safety] {arm[0]} ({len(todo)} prompts)")
        # Budget must clear this model's rumination: harmful prompts at temp 0
        # produce 3000+ tokens of thinking before the answer. 2048 truncated the
        # answer to empty and made compliances look like refusals. 16k leaves
        # room for a full think + a full answer.
        with Server(arm, mtp=arm[3], ctx=32768):
            for kind, x in todo:
                r = chat([{"role": "user", "content": x["prompt"]}], 16384,
                         sampling=SAFETY_SAMPLING)
                verdict = classify_safety(r)
                tf = os.path.join(TRANS, f"{arm[0]}_{kind}_{x['id']}.txt")
                with open(tf, "w", encoding="utf-8") as fh:
                    fh.write(f"### PROMPT\n{x['prompt']}\n\n### VERDICT {verdict}\n\n"
                             f"### THINKING\n{r['reasoning']}\n\n### ANSWER\n{r['content']}\n")
                print(f"    {kind[:1].upper()} {x['id']:14s} {verdict}"
                      f"{' [len]' if r['finish']=='length' else ''}")
                emit(dict(key=f"s|{arm[0]}|{kind}|{x['id']}", stage="safety", arm=arm[0],
                          kind=kind, item=x["id"], cat=x.get("cat"),
                          verdict=verdict,
                          refused=(verdict == "refuse"),
                          complied=(verdict == "comply"),
                          truncated=(r["finish"] == "length"),
                          out_chars=len(r["content"]), think_chars=len(r["reasoning"]),
                          finish=r["finish"]))


def stage_speed(done, redo):
    for arm in ARMS:
        if not ready(arm[1]):
            print(f"  skip {arm[0]}: not ready"); continue
        for mtp in (False, True):
            todo = [(c, rep) for c in SPEED_PROMPTS for rep in range(SPEED_REPEATS)
                    if f"sp|{arm[0]}|{'mtp' if mtp else 'no'}|{c}|{rep}" not in done or redo]
            if not todo:
                continue
            print(f"[speed] {arm[0]} mtp={mtp}")
            with Server(arm, mtp=mtp, ctx=8192):
                chat([{"role": "user", "content": "hi"}], 32)  # warmup
                for content, rep in todo:
                    r = chat([{"role": "user", "content": SPEED_PROMPTS[content]}],
                             SPEED_MAX_TOKENS)
                    tps = round((r["completion_tokens"] or 0) / r["wall_s"], 2)
                    acc = (round(r["draft_accepted"] / r["draft_n"], 4)
                           if r.get("draft_n") else None)
                    emit(dict(key=f"sp|{arm[0]}|{'mtp' if mtp else 'no'}|{content}|{rep}",
                              stage="speed", arm=arm[0], mtp=mtp, content=content, rep=rep,
                              client_tps=tps, accept=acc,
                              completion_tokens=r["completion_tokens"], wall_s=r["wall_s"]))
                    print(f"    {content} r{rep} {tps} tok/s accept={acc}")


def stage_ppl(done, redo):
    for arm in ARMS:
        if not ready(arm[1]):
            print(f"  skip {arm[0]}: not ready"); continue
        key = f"ppl|{arm[0]}"
        if key in done and not redo:
            continue
        print(f"[ppl] {arm[0]}")
        log = os.path.join(LOGS, f"ppl_{arm[0]}.log")
        cmd = [PPL_BIN, "-m", arm[1], "-f", CORPUS, "-ngl", "99", "-c", "512",
               "-fa", "on", "--no-warmup"]
        with open(log, "w", encoding="utf-8") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=TOOLS,
                           timeout=7200)
        txt = open(log, encoding="utf-8", errors="replace").read()
        m = re.findall(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", txt)
        ppl, err = (float(m[-1][0]), float(m[-1][1])) if m else (None, None)
        print(f"    PPL = {ppl} +/- {err}")
        emit(dict(key=key, stage="ppl", arm=arm[0], ppl=ppl, ppl_err=err))


STAGES = {"quality": stage_quality, "safety": stage_safety,
          "safety_nothink": stage_safety_nothink,
          "speed": stage_speed, "ppl": stage_ppl}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    redo = "--redo" in sys.argv
    if not args or args == ["all"]:
        args = ["quality", "safety", "speed", "ppl"]
    os.makedirs(LOGS, exist_ok=True)
    for a in args:
        if a not in STAGES:
            sys.exit(f"unknown stage {a}; pick {list(STAGES)}")
    for a in args:
        print(f"\n===== stage {a} =====")
        STAGES[a](load_done(), redo)
    print("\ndone")


if __name__ == "__main__":
    main()

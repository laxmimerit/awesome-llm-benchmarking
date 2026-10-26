"""Perplexity on wikitext-2 test, for every model in the comparison.

Valid as a cross-model comparison only because Flash-Next and Qwen3.8-27B were
verified to share a byte-identical tokenizer (vocab 248320, gpt2 BPE, pre=qwen35,
same BOS/EOS). Perplexity across different tokenizers would be meaningless.
"""
import argparse, json, os, re, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "research", "build-cuda", "bin", "llama-perplexity.exe")
SIB = os.path.abspath(os.path.join(ROOT, "..", "qwen38-unsloth-dynamic3", "models"))
CORPUS = os.path.abspath(os.path.join(ROOT, "..", "qwen38-uncensored-vs-base", "corpus", "wikitext2-test.raw"))
PLE = "per_layer_token_embd"

def fn_extra(ncmoe):
    return ["-ngl", "99", "-ncmoe", str(ncmoe), "-ot", PLE + "=CPU"]
MODELS = {
    "flashnext-IQ1_S":    (os.path.join(ROOT, "models", "UD-IQ1_S", "Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf"), fn_extra(20)),
    "flashnext-Q2_K_XL":  (os.path.join(ROOT, "models", "UD-Q2_K_XL", "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf"), fn_extra(28)),
    "qwen38-27b-Q4_K_M":  (os.path.join(SIB, "Qwen3.8-27B-UD-Q4_K_M.gguf"), ["-ngl", "99"]),
    "qwen38-27b-Q2_K_XL": (os.path.join(SIB, "Qwen3.8-27B-UD-Q2_K_XL.gguf"), ["-ngl", "99"]),
    "qwen38-27b-IQ1_S":   (os.path.join(SIB, "Qwen3.8-27B-UD-IQ1_S.gguf"), ["-ngl", "99"]),
}

FINAL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--chunks", type=int, default=40)
    ap.add_argument("--ctx", default="2048")
    a = ap.parse_args()

    out = os.path.join(ROOT, "results", "ppl.jsonl")
    logs = os.path.join(ROOT, "logs")
    print("corpus:", CORPUS, os.path.getsize(CORPUS), "bytes")

    for mk in a.models.split(","):
        path, extra = MODELS[mk]
        if not os.path.exists(path):
            print("SKIP %s: missing %s" % (mk, path), flush=True)
            continue
        cmd = [BIN, "-m", path, "-f", CORPUS, "-c", a.ctx, "--chunks", str(a.chunks),
               "-t", "24", "--no-warmup"] + extra
        print("\n########## %s\n%s" % (mk, " ".join(cmd)), flush=True)
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        blob = p.stdout + "\n" + p.stderr
        with open(os.path.join(logs, "ppl_%s.log" % mk), "w", encoding="utf-8") as fh:
            fh.write("CMD: " + " ".join(cmd) + "\n\n" + blob)
        m = FINAL.search(blob)
        rec = dict(model=mk, chunks=a.chunks, ctx=int(a.ctx), wall_s=round(dt, 1),
                   exit=p.returncode, cmd=cmd,
                   ppl=float(m.group(1)) if m else None,
                   stderr_ppl=float(m.group(2)) if m else None)
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if m:
            print("   PPL = %s +/- %s   (%.0fs)" % (m.group(1), m.group(2), dt), flush=True)
        else:
            print("   NO PPL PARSED exit=%d\n%s" % (p.returncode, blob[-1500:]), flush=True)


if __name__ == "__main__":
    main()

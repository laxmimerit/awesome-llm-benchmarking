"""What abliteration actually changed, measured at the GGUF weight level.

Three files, two deltas:

    A  stock Q4_K_M            (Ollama blob, unmodified Qwen 3.8 27B)
    B  uncensored Q4_K_M       (orcarouter, abliterated)
    C  Unsloth UD-Q4_K_M       (unmodified base, DIFFERENT quantizer)

B-A carries quantizer difference PLUS whatever abliteration did.
C-A carries quantizer difference ONLY. C is the noise floor: any structure in
B-A that is absent from C-A cannot be a quantization artifact.

Per tensor we record:
  rel        ||delta||_F / ||W||_F
  top1       sigma_1^2 / ||delta||_F^2, the share of the delta explained by its
             single strongest rank-1 component. A residual-stream ablation
             delta = r r^T W is exactly rank 1, so top1 -> 1.0. Quantization
             error is full-rank white noise, so top1 -> ~1/min(shape).
  u1         the leading singular vector on the n_embd (residual) axis.

The decisive test is cross-layer agreement of u1. One shared refusal direction
orthogonalized out of every layer means every modified tensor's u1 is the same
unit vector up to sign. Quantization noise gives unrelated directions.

Usage:
  python ggufdiff.py structure      # metadata + tensor inventory, exact
  python ggufdiff.py weights        # per-tensor delta + rank, appends jsonl
  python ggufdiff.py direction      # cross-layer u1 cosine matrix
  python ggufdiff.py all
"""
import json, os, re, sys
import numpy as np
from gguf import GGUFReader, dequantize

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "diff.jsonl")
DIRS = os.path.join(BASE, "directions.npz")

A_PATH = ("E:/AI/ollama/blobs/"
          "sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d")
B_PATH = os.path.join(BASE, "models", "Qwen3.8-27B-Uncensored-Q4_K_M.gguf")
C_PATH = os.path.join(os.path.dirname(BASE), "qwen38-unsloth-dynamic3",
                      "models", "Qwen3.8-27B-UD-Q4_K_M.gguf")

N_EMBD = 5120
B_EXPECTED = 16810714496  # exact upstream byte size, so a partial file is never used


def readers(need_c=True):
    for p in (A_PATH, B_PATH) + ((C_PATH,) if need_c else ()):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")
    if os.path.getsize(B_PATH) != B_EXPECTED:
        sys.exit(f"B is {os.path.getsize(B_PATH)} bytes, expected {B_EXPECTED}; "
                 "download still running")
    out = [GGUFReader(A_PATH), GGUFReader(B_PATH)]
    if need_c:
        out.append(GGUFReader(C_PATH))
    return out


def fam(name):
    """Collapse blk.31.ffn_down.weight -> ffn_down."""
    m = re.match(r"blk\.\d+\.(.+)$", name)
    return m.group(1).replace(".weight", "") if m else name


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


# tensors that write into the residual stream: the only place an ablation can act
RESIDUAL = {"attn_output", "ssm_out", "ffn_down"}


# ------------------------------------------------------------------- structure

def stage_structure():
    ra, rb, rc = readers()
    print("=" * 72)
    print("METADATA")
    print("=" * 72)
    ka, kb = set(ra.fields), set(rb.fields)
    print(f"fields: A={len(ka)} B={len(kb)}  only-in-A={sorted(ka - kb)}  "
          f"only-in-B={sorted(kb - ka)}")
    diffs = 0
    for k in sorted(ka & kb):
        try:
            va, vb = ra.fields[k].contents(), rb.fields[k].contents()
        except Exception:
            continue
        sa, sb = str(va), str(vb)
        if sa != sb:
            diffs += 1
            print(f"  DIFFERS {k}")
            print(f"    A: {sa[:300]}")
            print(f"    B: {sb[:300]}")
    print(f"  -> {diffs} differing metadata fields out of {len(ka & kb)}")

    print("\n" + "=" * 72)
    print("TENSOR INVENTORY")
    print("=" * 72)
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    print(f"count: A={len(ta)} B={len(tb)}")
    print(f"only-in-A: {sorted(set(ta) - set(tb))[:20]}")
    print(f"only-in-B: {sorted(set(tb) - set(ta))[:20]}")
    shape_mismatch, dtype_mismatch = [], []
    for n in sorted(set(ta) & set(tb)):
        if tuple(ta[n].shape) != tuple(tb[n].shape):
            shape_mismatch.append((n, tuple(ta[n].shape), tuple(tb[n].shape)))
        if ta[n].tensor_type != tb[n].tensor_type:
            dtype_mismatch.append((n, ta[n].tensor_type.name, tb[n].tensor_type.name))
    print(f"shape mismatches: {len(shape_mismatch)}")
    for x in shape_mismatch[:10]:
        print("   ", x)
    print(f"quant-type mismatches: {len(dtype_mismatch)}")
    from collections import Counter
    cc = Counter((a, b) for _, a, b in dtype_mismatch)
    for (a, b), n in cc.most_common():
        print(f"    {n:4d}  A={a} -> B={b}")


# --------------------------------------------------------------------- weights

def as_embd_major(arr):
    """Return the matrix oriented so axis 0 is the n_embd (residual) axis.

    Returns None when the tensor has no n_embd axis or is not 2-D.
    """
    if arr.ndim != 2:
        return None
    if arr.shape[0] == N_EMBD:
        return arr
    if arr.shape[1] == N_EMBD:
        return arr.T
    return None


def top1_share(d, iters=30, seed=0):
    """sigma_1^2 / ||d||_F^2 via power iteration on d d^T. d is (m, n) float32."""
    fro2 = float(np.sum(d.astype(np.float64) ** 2))
    if fro2 == 0:
        return 0.0, None, 0.0
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(d.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = d @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0, None, fro2
        u /= nu
        v = d.T @ u
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0, None, fro2
        v /= nv
    s1 = float(np.linalg.norm(d @ v))
    return s1 ** 2 / fro2, u.astype(np.float32), fro2


def stage_weights():
    ra, rb, rc = readers()
    ta = {t.name: t for t in ra.tensors}
    tb = {t.name: t for t in rb.tensors}
    tc = {t.name: t for t in rc.tensors}
    names = [n for n in ta if n in tb]
    names.sort(key=lambda n: (layer_of(n), n))

    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(line)["name"])
            except Exception:
                pass

    dirs = {}
    if os.path.exists(DIRS):
        dirs = dict(np.load(DIRS))

    fh = open(OUT, "a", encoding="utf-8")
    for i, n in enumerate(names):
        if n in done:
            continue
        try:
            wa = dequantize(ta[n].data, ta[n].tensor_type).astype(np.float32)
            wb = dequantize(tb[n].data, tb[n].tensor_type).astype(np.float32)
        except Exception as e:
            print(f"  skip {n}: {type(e).__name__} {e}")
            continue
        if wa.shape != wb.shape:
            continue
        na = float(np.linalg.norm(wa))
        dab = wb - wa
        rec = dict(name=n, fam=fam(n), layer=layer_of(n), shape=list(wa.shape),
                   qa=ta[n].tensor_type.name, qb=tb[n].tensor_type.name,
                   norm_a=na,
                   rel_ab=float(np.linalg.norm(dab)) / na if na else None)

        # control delta against the third quantizer of the same base weights
        if n in tc:
            try:
                wc = dequantize(tc[n].data, tc[n].tensor_type).astype(np.float32)
                if wc.shape == wa.shape:
                    dac = wc - wa
                    rec["qc"] = tc[n].tensor_type.name
                    rec["rel_ac"] = float(np.linalg.norm(dac)) / na if na else None
            except Exception:
                dac = None
            else:
                dac = dac if wc.shape == wa.shape else None
        else:
            dac = None

        # rank structure, only where a residual axis exists
        m = as_embd_major(dab)
        if m is not None:
            share, u1, _ = top1_share(m)
            rec["top1_ab"] = share
            if u1 is not None:
                dirs[n] = u1
        if dac is not None:
            mc = as_embd_major(dac)
            if mc is not None:
                rec["top1_ac"] = top1_share(mc)[0]

        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if i % 25 == 0:
            print(f"  [{i}/{len(names)}] {n} rel_ab={rec['rel_ab']:.5f} "
                  f"rel_ac={rec.get('rel_ac', float('nan')):.5f} "
                  f"top1_ab={rec.get('top1_ab', float('nan')):.4f}")
    fh.close()
    if dirs:
        np.savez_compressed(DIRS, **dirs)
    print(f"wrote {OUT} and {DIRS}")


# ------------------------------------------------------------------- direction

def _cos_stats(M):
    """Off-diagonal |cos| stats for a stack of unit row vectors."""
    C = np.abs(M @ M.T)
    off = C[~np.eye(len(C), dtype=bool)]
    return dict(n=len(M), mean=float(off.mean()), median=float(np.median(off)),
                mn=float(off.min()), mx=float(off.max()))


def stage_direction():
    """Group by tensor ROLE, not by delta magnitude.

    An earlier version gated on an 8x-the-noise-floor threshold and so dropped
    33 ffn_down tensors whose edit is real but small (the ablation tapers at
    the first and last blocks). Those dropped tensors align to the consensus at
    |cos| 0.9999, i.e. the threshold was hiding the cleanest evidence. Role is
    the correct split: a residual writer can carry the ablation, an input
    projection cannot.
    """
    if not os.path.exists(DIRS):
        sys.exit("run `python ggufdiff.py weights` first")
    d = np.load(DIRS)
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    by = {r["name"]: r for r in recs}

    inp_fams = {"attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate",
                "ffn_gate", "ffn_up"}

    def pick(pred):
        ns = [n for n in d.files
              if pred(by.get(n, {})) and np.linalg.norm(d[n]) > 0]
        ns.sort(key=lambda n: (by[n]["fam"], by[n]["layer"]))
        return ns

    resid = pick(lambda b: b.get("fam") in RESIDUAL)
    embd = [n for n in ("token_embd.weight",)
            if n in d.files and np.linalg.norm(d[n]) > 0]
    control = pick(lambda b: b.get("fam") in inp_fams)
    edited = resid + embd

    print(f"residual writers: {len(resid)}   + token_embd: {len(embd)}   "
          f"-> edited set {len(edited)}")
    print(f"control (input projections, cannot carry an ablation): {len(control)}")

    def unit_stack(ns):
        return np.stack([d[n] / np.linalg.norm(d[n]) for n in ns])

    Me = unit_stack(edited)
    u, s, vt = np.linalg.svd(Me, full_matrices=False)
    r = vt[0]
    align = np.abs(Me @ r)
    rc = np.sqrt(2 / (np.pi * N_EMBD))

    print("")
    print(f"consensus direction holds {s[0]**2/np.sum(s**2)*100:.2f}% of the "
          f"stacked-direction energy")
    print(f"EDITED   |cos| mean {align.mean():.5f}  median "
          f"{np.median(align):.5f}  min {align.min():.5f}")

    from collections import defaultdict
    g = defaultdict(list)
    for n, a in zip(edited, align):
        g[by[n]["fam"]].append(a)
    print("")
    print("per-family alignment to the consensus direction:")
    for f, v in sorted(g.items(), key=lambda kv: -len(kv[1])):
        print(f"  {f:18s} n={len(v):3d}  mean |cos| {np.mean(v):.5f}  "
              f"min {np.min(v):.5f}")

    Mc = unit_stack(control)
    ca = np.abs(Mc @ r)
    print("")
    print(f"CONTROL  |cos| mean {ca.mean():.5f}  max {ca.max():.5f}")
    print(f"random-chance baseline for d={N_EMBD}: {rc:.5f}")
    print(f"  -> control sits at {ca.mean()/rc:.2f}x chance (1.00 = pure noise)")

    np.save(os.path.join(BASE, "refusal_direction.npy"), r)
    print("saved refusal_direction.npy")


STAGES = {"structure": stage_structure, "weights": stage_weights,
          "direction": stage_direction}

if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    if args == ["all"]:
        args = ["structure", "weights", "direction"]
    for a in args:
        if a not in STAGES:
            sys.exit(f"unknown stage {a}; pick from {list(STAGES)}")
        print(f"\n##### {a} #####")
        STAGES[a]()

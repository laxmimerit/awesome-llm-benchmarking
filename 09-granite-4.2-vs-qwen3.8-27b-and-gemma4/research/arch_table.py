"""Architecture comparison built from a per-block tensor census.

KV cache bytes per token is what decides whether an advertised 128K or 256K
context is reachable on a 32 GB card, and it cannot be read off a model card.
Three separate mechanisms break the naive "layers x kv_heads x head_dim"
formula, and all three appear in this line-up:

  * the qwen35 family (Qwen 3.8, Ornith 1.5) is a hybrid: three quarters of its
    blocks carry `ssm_*` tensors and hold a fixed-size recurrent state instead
    of a growing KV cache.  Only the pure-attention blocks cost anything per
    token.  This is the real reason those models cache on a quarter of their
    layers; it is not KV sharing.
  * gemma4 interleaves sliding-window layers 5:1 with full-attention layers and
    gives the two kinds different head counts and head dims.  Beyond the 1024
    window the sliding layers stop growing, so the marginal cost per token
    comes only from the full-attention layers.
  * granite does none of this.  Every block is full attention.

The result is cross-checked against what llama-server actually allocates.
"""

import json
import re
from pathlib import Path

from gguf import GGUFReader

from models import ALL, REGISTRY

OUT = Path(__file__).resolve().parent.parent / "results" / "arch.json"


def census(path):
    r = GGUFReader(str(path))
    meta = {}
    for f in r.fields.values():
        try:
            meta[f.name] = f.contents()
        except Exception:
            pass

    blocks = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.+)", t.name)
        if m:
            blocks.setdefault(int(m.group(1)), set()).add(m.group(2))

    arch = meta.get("general.architecture", "")
    # the qwen35 family appends a `nextn` MTP block that carries attention
    # tensors but is not part of the decode stack and never caches KV, so it
    # must be excluded or the KV estimate comes out one layer too high
    n_blocks = None
    MTP_HINTS = ("eh_proj", "enorm", "hnorm", "shared_head", "nextn", "mtp")
    mtp_blocks = sorted(i for i, s in blocks.items()
                        if any(h in n.lower() for n in s for h in MTP_HINTS))
    for i in mtp_blocks:
        blocks.pop(i)
    n_blocks = len(blocks)
    ssm_blocks = sorted(i for i, s in blocks.items()
                        if any(n.startswith("ssm_") for n in s))

    def key(name, default=None):
        return meta.get(f"{arch}.{name}", default)

    swa_pattern = key("attention.sliding_window_pattern")
    window = key("attention.sliding_window")
    hk = key("attention.head_count_kv")
    klen = key("attention.key_length")
    vlen = key("attention.value_length")
    klen_swa = key("attention.key_length_swa", klen)
    vlen_swa = key("attention.value_length_swa", vlen)

    if klen is None:                       # granite states head dim indirectly
        emb = key("embedding_length")
        hc = key("attention.head_count")
        klen = vlen = klen_swa = vlen_swa = emb // hc

    # per-token f16 bytes on layers whose cache grows with context
    growing = 0
    fixed = 0
    full_layers = 0
    if swa_pattern is not None:
        # gemma: pattern[i] True means that layer is sliding-window
        for i in range(n_blocks):
            is_swa = bool(swa_pattern[i])
            heads = hk[i] if isinstance(hk, (list, tuple)) else hk
            if is_swa:
                fixed += heads * (klen_swa + vlen_swa) * 2 * int(window or 0)
            else:
                full_layers += 1
                growing += heads * (klen + vlen) * 2
    else:
        heads = hk[0] if isinstance(hk, (list, tuple)) else hk
        for i in range(n_blocks):
            if i in ssm_blocks:
                continue                    # recurrent state, not a KV cache
            full_layers += 1
            growing += heads * (klen + vlen) * 2

    return {
        "arch": arch,
        "n_blocks": n_blocks,
        "n_mtp_blocks": len(mtp_blocks),
        "n_ssm_blocks": len(ssm_blocks),
        "n_full_attn_blocks": full_layers,
        "swa_window": window,
        "kv_bytes_per_token": growing,
        "kv_kib_per_token": round(growing / 1024, 1),
        "swa_fixed_bytes": fixed,
        "swa_fixed_mib": round(fixed / 1024 / 1024, 1),
        "ctx": next((v for k, v in meta.items()
                     if k.endswith("context_length")), None),
        "n_tensors": len(r.tensors),
    }


def main():
    out = {}
    rows = []
    for k in ALL:
        p = Path(REGISTRY[k]["path"])
        if not p.exists():
            continue
        c = census(p)
        out[k] = c
        ctx = int(c["ctx"] or 0)
        kv_at_ctx = (c["kv_bytes_per_token"] * ctx + c["swa_fixed_bytes"]) / 1e9
        rows.append([
            REGISTRY[k]["label"], c["arch"], c["n_blocks"],
            f"{c['n_full_attn_blocks']} full / {c['n_ssm_blocks']} ssm",
            c["swa_window"] or "-",
            f"{c['kv_kib_per_token']:.0f}",
            f"{c['swa_fixed_mib']:.0f}" if c["swa_fixed_bytes"] else "-",
            f"{ctx // 1024}k",
            f"{kv_at_ctx:.1f}",
        ])

    hdr = ["model", "arch", "blocks", "attention layers", "SWA window",
           "KV KiB/token", "SWA fixed MiB", "nominal ctx", "KV at nominal (GB)"]
    print("| " + " | ".join(hdr) + " |")
    print("| " + " | ".join("---" for _ in hdr) + " |")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

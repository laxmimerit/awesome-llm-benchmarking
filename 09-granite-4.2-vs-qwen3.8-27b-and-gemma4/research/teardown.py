"""Architecture teardown for every model in the Ornith benchmark.

Dumps GGUF metadata + a tensor census to JSON so the "what is actually
different about Ornith" question is answered from the file, not the card.

Usage:
    python teardown.py <name>=<path.gguf> [<name>=<path.gguf> ...]
    python teardown.py --all          # uses models.py registry
"""

import json
import sys
from collections import Counter
from pathlib import Path

from gguf import GGUFReader

OUT = Path(__file__).resolve().parent.parent / "results" / "teardown.json"

# Tensor-name substrings that betray a speculative / multi-token-prediction head.
MTP_HINTS = ("mtp", "nextn", "draft", "eh_proj", "enorm", "hnorm", "shared_head")

# Metadata keys worth pulling out by name. Anything matching these substrings
# is captured; the rest of the (large) KV store is summarised only.
WANTED = (
    "general.architecture",
    "general.name",
    "general.basename",
    "general.size_label",
    "general.file_type",
    "general.quantization_version",
    "block_count",
    "context_length",
    "embedding_length",
    "feed_forward_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "attention.value_length",
    "attention.layer_norm_rms_epsilon",
    "attention.sliding_window",
    "rope.freq_base",
    "rope.scaling",
    "rope.dimension_count",
    "expert_count",
    "expert_used_count",
    "expert_shared_count",
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "nextn",
    "mtp",
    "vocab_size",
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.add_bos_token",
)


def field_value(field):
    try:
        val = field.contents()
    except Exception:
        return "<unreadable>"
    if isinstance(val, bytes):
        try:
            val = val.decode("utf-8", "replace")
        except Exception:
            val = str(val)
    return val


def profile(name, path):
    path = Path(path)
    r = GGUFReader(str(path))

    meta = {}
    for field in r.fields.values():
        if any(w in field.name for w in WANTED):
            val = field_value(field)
            if isinstance(val, str) and len(val) > 200:
                val = val[:200] + f"...<{len(val)} chars>"
            meta[field.name] = val

    # Chat template is captured separately and in full: the Ornith card says a
    # MODIFIED Qwen template is required, so the embedded one must be checked.
    template = None
    for field in r.fields.values():
        if field.name == "tokenizer.chat_template":
            template = field_value(field)

    tensors = r.tensors
    dtypes = Counter(t.tensor_type.name for t in tensors)

    blocks = set()
    for t in tensors:
        parts = t.name.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
            blocks.add(int(parts[1]))

    mtp_tensors = [t for t in tensors if any(h in t.name.lower() for h in MTP_HINTS)]

    # Per-block tensor-name shape, with the block index stripped, so two models
    # can be compared structurally regardless of depth.
    def shape_key(t):
        parts = t.name.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
            parts[1] = "N"
        return ".".join(parts)

    block0 = sorted(
        {(shape_key(t), tuple(int(x) for x in t.shape), t.tensor_type.name)
         for t in tensors
         if t.name.startswith("blk.0.")}
    )

    n_params = sum(int(1) * int(_prod(t.shape)) for t in tensors)

    return {
        "name": name,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_size_gb": round(path.stat().st_size / 1e9, 2),
        "n_tensors": len(tensors),
        "n_params_est": n_params,
        "n_params_est_b": round(n_params / 1e9, 2),
        "dtypes": dict(dtypes),
        "block_min": min(blocks) if blocks else None,
        "block_max": max(blocks) if blocks else None,
        "block_count_actual": len(blocks),
        "metadata": meta,
        "chat_template": template,
        "chat_template_len": len(template) if template else 0,
        "mtp_tensor_count": len(mtp_tensors),
        "mtp_tensors": [
            {"name": t.name, "shape": [int(x) for x in t.shape], "dtype": t.tensor_type.name}
            for t in mtp_tensors
        ],
        "block0_layout": [
            {"name": n, "shape": list(s), "dtype": d} for n, s, d in block0
        ],
    }


def _prod(shape):
    out = 1
    for x in shape:
        out *= int(x)
    return out


def main(argv):
    pairs = []
    for arg in argv:
        if "=" not in arg:
            print(f"skipping malformed arg: {arg}")
            continue
        name, path = arg.split("=", 1)
        pairs.append((name, path))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text())

    for name, path in pairs:
        if not Path(path).exists():
            print(f"MISSING: {name} -> {path}")
            continue
        print(f"profiling {name} ...", flush=True)
        existing[name] = profile(name, path)
        p = existing[name]
        print(
            f"  arch={p['metadata'].get('general.architecture')} "
            f"tensors={p['n_tensors']} blocks={p['block_count_actual']} "
            f"params~{p['n_params_est_b']}B mtp_tensors={p['mtp_tensor_count']}"
        )

    OUT.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])

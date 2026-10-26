"""Inspect a GGUF file for MTP (multi-token-prediction) draft-head tensors.

Qwen3.8 ships an MTP head inside the GGUF. llama.cpp's --spec-type draft-mtp
only works if those tensors survived the quantisation/conversion. This prints
the architecture metadata and any tensor that looks like an MTP / draft head.

Usage: python inspect_gguf.py <path-to.gguf>
"""

import sys

from gguf import GGUFReader

MTP_HINTS = ("mtp", "nextn", "draft", "eh_proj", "enorm", "hnorm", "shared_head")


def main(path):
    print(f"Reading: {path}\n")
    r = GGUFReader(path)

    print("=== key metadata ===")
    wanted = (
        "general.architecture",
        "general.name",
        "general.file_type",
        "block_count",
        "context_length",
        "nextn",
        "mtp",
        "num_nextn",
    )
    for field in r.fields.values():
        name = field.name
        if any(w in name for w in wanted):
            try:
                val = field.contents()
            except Exception:
                val = "<unreadable>"
            if isinstance(val, str) and len(val) > 80:
                val = val[:80] + "..."
            print(f"  {name} = {val}")

    print(f"\n=== tensors: {len(r.tensors)} total ===")
    mtp = [t for t in r.tensors if any(h in t.name.lower() for h in MTP_HINTS)]
    if mtp:
        print(f"  MTP / draft-head tensors found: {len(mtp)}")
        for t in mtp:
            print(f"    {t.name:<48} {str(list(t.shape)):<22} {t.tensor_type.name}")
    else:
        print("  !! NO MTP / draft-head tensors found")

    # Highest block index tells us if there is an extra layer past block_count.
    blocks = set()
    for t in r.tensors:
        parts = t.name.split(".")
        if len(parts) > 2 and parts[0] == "blk" and parts[1].isdigit():
            blocks.add(int(parts[1]))
    if blocks:
        print(f"\n  blk indices: {min(blocks)}..{max(blocks)}  (count={len(blocks)})")

    print(f"\n  VERDICT: {'MTP PRESENT' if mtp else 'MTP ABSENT'}")


if __name__ == "__main__":
    main(sys.argv[1])

"""Fetch safetensors shard headers via HTTP range requests.

Builds a complete {tensor_name: {dtype, shape}} map for a HF repo
without downloading the weights themselves.
"""
import json
import struct
import sys
import urllib.request

REPOS = {
    "glm53": ("zai-org/GLM-5.3", "models/glm53_index.json", "results/glm53_tensors.json"),
    "glm53_flash": ("zai-org/GLM-5.3-Flash", "models/glm53_flash_index.json", "results/glm53_flash_tensors.json"),
}


def fetch_range(url, start, end):
    req = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "teardown"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main(key):
    repo, index_path, out_path = REPOS[key]
    index = json.load(open(index_path))
    files = sorted(set(index["weight_map"].values()))
    tensors = {}
    for i, fn in enumerate(files):
        url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
        for attempt in range(3):
            try:
                n = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
                hdr = json.loads(fetch_range(url, 8, 7 + n))
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retry {fn}: {e}", flush=True)
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            tensors[name] = {"dtype": info["dtype"], "shape": info["shape"]}
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"{key}: {i+1}/{len(files)} shards, {len(tensors)} tensors", flush=True)
    json.dump(tensors, open(out_path, "w"))
    print(f"{key}: wrote {out_path} with {len(tensors)} tensors", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

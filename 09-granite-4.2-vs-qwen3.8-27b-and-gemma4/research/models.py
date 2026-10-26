"""Model registry for the Granite 4.2 benchmark.

Two brackets, six models, every one of them Q4_K_M so quant class is never a
confound.

  Bracket A ("what fits in about 8 GB"):  Granite 4.2 8B, Gemma 4 12B,
      Ornith 1.5 9B.  These are NOT parameter matched (8 / 12 / 9 B) and are
      reported as a VRAM bracket, not as a like for like size comparison.

  Bracket B ("matched dense ~30B"):  Granite 4.2 30B, Gemma 4 31B,
      Qwen 3.8 27B.  17.0 to 18.3 GB of weights, all dense, all reasoning
      models.  This is the clean comparison.

Paths point either at a GGUF already downloaded for an earlier project or at
the plain GGUF blob inside the Ollama store, which avoids re-downloading many
GB.  Feeding the blob to llama-server directly also means none of Ollama's
undocumented flag defaults (speculative decoding, chatml override, batch
sizes) come along for the ride.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

OLLAMA_ROOT = Path("E:/AI/ollama")
OLLAMA_BLOBS = OLLAMA_ROOT / "blobs"
OLLAMA_MANIFESTS = OLLAMA_ROOT / "manifests" / "registry.ollama.ai" / "library"

# GGUFs already on disk from the Ornith round.
ORNITH_MODELS = ROOT.parent / "ornith-benchmark" / "models"

# llama.cpp build used for every run in this project.  Set by bootstrap check;
# if b10448 does not know the Granite 4.2 architecture this points at a newer
# build and EVERY model is re-run on it, never mixed.
TOOLS = ROOT / "tools"
if not (TOOLS / "llama-server.exe").exists():
    TOOLS = ROOT.parent / "qwen3.8-mtp-benchmark" / "tools"
SERVER = TOOLS / "llama-server.exe"


def ollama_blob(model, tag):
    """Resolve an Ollama tag to the plain GGUF blob holding its weights."""
    man = OLLAMA_MANIFESTS / model / tag
    if not man.exists():
        return Path(f"<not pulled: {model}:{tag}>")
    data = json.loads(man.read_text())
    layers = [l for l in data["layers"]
              if l["mediaType"].endswith("model")]
    if not layers:
        return Path(f"<no model layer in {model}:{tag}>")
    digest = max(layers, key=lambda l: l["size"])["digest"].replace(":", "-")
    return OLLAMA_BLOBS / digest


def ollama_projector(model, tag):
    """mmproj blob for a vision model, if the tag ships one."""
    man = OLLAMA_MANIFESTS / model / tag
    if not man.exists():
        return None
    data = json.loads(man.read_text())
    for l in data["layers"]:
        if "projector" in l["mediaType"]:
            return OLLAMA_BLOBS / l["digest"].replace(":", "-")
    return None


REGISTRY = {
    # ---- Bracket A: about 8 GB of weights ------------------------------
    "granite4.2-8b": {
        "label": "Granite-4.2-8B",
        "path": ollama_blob("granite4.2", "8b-q4_K_M"),
        "family": "granite",
        "bracket": "A",
        # from the model's own Ollama Modelfile: temperature 1, top_p 0.95,
        # no top_k set
        "sampling": {"temperature": 1.0, "top_p": 0.95},
        "thinking": True,
    },
    "gemma4-12b": {
        "label": "Gemma-4-12B",
        "path": ollama_blob("gemma4", "12b-it-q4_K_M"),
        "family": "gemma",
        "bracket": "A",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "thinking": True,
    },
    "ornith-9b": {
        "label": "Ornith-1.5-9B",
        "path": ORNITH_MODELS / "Ornith-1.5-9B-Q4_K_M.gguf",
        "family": "ornith",
        "bracket": "A",
        # card: general tasks temperature=1.0, top_p=0.95, top_k=20,
        # min_p=0.0, presence_penalty=1.5
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 1.5},
        "thinking": True,
    },

    # ---- Bracket B: matched dense ~30B ---------------------------------
    "granite4.2-30b": {
        "label": "Granite-4.2-30B",
        "path": ollama_blob("granite4.2", "30b-q4_K_M"),
        "family": "granite",
        "bracket": "B",
        "sampling": {"temperature": 1.0, "top_p": 0.95},
        "thinking": True,
    },
    "gemma4-31b": {
        "label": "Gemma-4-31B",
        "path": ORNITH_MODELS / "gemma-4-31B-it-Q4_K_M.gguf",
        "family": "gemma",
        "bracket": "B",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "thinking": True,
    },
    "qwen3.8-27b": {
        "label": "Qwen3.8-27B",
        "path": ollama_blob("qwen3.8", "latest"),
        "family": "qwen",
        "bracket": "B",
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "presence_penalty": 0.0},
        "thinking": True,
    },
}

BRACKET_A = ["granite4.2-8b", "gemma4-12b", "ornith-9b"]
BRACKET_B = ["granite4.2-30b", "gemma4-31b", "qwen3.8-27b"]
ALL = BRACKET_A + BRACKET_B

# Granite 4.2 advertises configurable reasoning effort.  Qwen 3.8 advertised
# the same and Ollama never set the template variable, so the dial was a no-op.
# Tested here through llama-server --jinja with the model's own template.
GRANITE = ["granite4.2-8b", "granite4.2-30b"]


def available():
    return [k for k in ALL if Path(REGISTRY[k]["path"]).exists()]


if __name__ == "__main__":
    for k in ALL:
        p = Path(REGISTRY[k]["path"])
        ok = "OK " if p.exists() else "MISSING"
        size = f"{p.stat().st_size/1e9:6.2f} GB" if p.exists() else "        "
        print(f"{ok} {k:16s} {size}  {p}")
    print(f"\nserver: {SERVER}  exists={SERVER.exists()}")

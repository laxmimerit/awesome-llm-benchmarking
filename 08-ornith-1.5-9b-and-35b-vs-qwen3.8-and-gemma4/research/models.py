"""Model registry for the Ornith 1.5 benchmark.

Every model is Q4_K_M so quant class is never a confound. Paths point either at
the downloaded HF GGUF or at the plain-GGUF blob inside the Ollama store.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OLLAMA_BLOBS = Path("E:/AI/ollama/blobs")

# llama.cpp build used for every run in this project.
LLAMA_BIN = (
    ROOT.parent / "qwen3.8-mtp-benchmark" / "tools"
)
SERVER = LLAMA_BIN / "llama-server.exe"

REGISTRY = {
    "ornith-9b": {
        "label": "Ornith-1.5-9B",
        "path": MODELS / "Ornith-1.5-9B-Q4_K_M.gguf",
        "family": "ornith",
        "kind": "dense",
        # Card: general tasks temperature=1.0, top_p=0.95, top_k=20,
        # min_p=0.0, presence_penalty=1.5
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 1.5},
        # Card's coding profile, used for the presence_penalty A/B.
        "sampling_alt": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                         "min_p": 0.0, "presence_penalty": 0.0},
        "thinking": True,
    },
    "ornith-35b-a3b": {
        "label": "Ornith-1.5-35B-A3B",
        "path": MODELS / "Ornith-1.5-35B-Q4_K_M.gguf",
        "family": "ornith",
        "kind": "moe",
        # Card: general tasks temperature=0.6, top_p=0.95, top_k=20
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "sampling_alt": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "thinking": True,
    },
    "qwen3.8-27b": {
        "label": "Qwen3.8-27B",
        "path": OLLAMA_BLOBS / "sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d",
        "family": "qwen",
        "kind": "dense+mtp",
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                     "presence_penalty": 0.0},
        "thinking": True,
        "note": "paired anchor only, not a full re-run of the earlier teardown",
    },
    # Gemma 4-31B was benchmarked first because Ornith's card names it by name.
    # It is dense, and the only axis it existed to test (agentic coding) came
    # back saturated, so it was replaced by the 26B: a 128-expert mixture, which
    # is the like-for-like architectural rival to Ornith 35B-A3B and also the
    # Gemma most people actually have pulled. Kept here for reference.
    "gemma4-31b": {
        "label": "Gemma-4-31B",
        "path": MODELS / "gemma-4-31B-it-Q4_K_M.gguf",
        "family": "gemma",
        "kind": "dense",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "thinking": True,
        "note": "superseded by gemma4-26b, not in the published field",
    },
    # The Ollama blob for gemma4:26b packs the vision tower into the same file
    # (655 text + 355 vision tensors + projector). llama.cpp loads the 658 text
    # tensors and then fails its total-count check, so a text-only build is used
    # instead. Plain Q4_K_M, not Unsloth Dynamic, so quant class stays matched.
    "gemma4-26b": {
        "label": "Gemma-4-26B-A4B",
        "path": MODELS / "gemma-4-26B-A4B-it-Q4_K_M.gguf",
        "family": "gemma",
        "kind": "moe",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "thinking": True,
    },
    "muse-glimmer-30b": {
        "label": "Muse-Glimmer-30B",
        "path": OLLAMA_BLOBS / "sha256-71b5c9c9abbc2a9e7c13550b6f9fe1f062650117a24490b15a310cbd7de03f8b",
        "family": "muse",
        "kind": "dense+swa",
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "thinking": True,
    },
}

# The two models the Ornith card makes head-to-head claims against, plus the
# rig's standing anchor.
# Field trimmed to four so the article stays readable. Muse Glimmer 30B and
# Gemma 4-31B were both measured in the v1 sweep and are kept in the archive.
CARD_RIVALS = ["qwen3.8-27b", "gemma4-26b"]
ORNITH = ["ornith-9b", "ornith-35b-a3b"]
ALL = ORNITH + CARD_RIVALS


def available():
    return [k for k in ALL if REGISTRY[k]["path"].exists()]


def missing():
    return [k for k in ALL if not REGISTRY[k]["path"].exists()]


if __name__ == "__main__":
    for k in ALL:
        m = REGISTRY[k]
        ok = m["path"].exists()
        size = f"{m['path'].stat().st_size/1e9:.2f} GB" if ok else "MISSING"
        print(f"{k:20} {m['kind']:12} {size:>12}  {m['path']}")
    print("\nserver:", SERVER, "exists" if SERVER.exists() else "MISSING")

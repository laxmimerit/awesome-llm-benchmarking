"""Architecture teardown of GLM-5.3 vs GLM-5.3-Flash.

Inputs (raw sources of truth):
  models/glm53_config.json, models/glm53_flash_config.json, models/glm5_config.json
  results/glm53_tensors.json, results/glm53_flash_tensors.json  (full shape maps
  fetched from every safetensors shard header via HTTP range requests)
  models/*_index.json (weight maps, total_size)

Outputs:
  results/architecture_analysis.json plus a printed summary.
"""
import json
import math
import re
from collections import defaultdict

BYTES = {"F8_E4M3": 1, "BF16": 2, "F32": 4, "F16": 2}


def load(p):
    return json.load(open(p))


def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def group_tensors(tensors, prefix_strip=""):
    """Group tensor names into components; return {component: {params, bytes, count}}."""
    groups = defaultdict(lambda: {"params": 0, "bytes": 0, "tensors": 0})
    for name, info in tensors.items():
        n = numel(info["shape"])
        b = n * BYTES[info["dtype"]]
        g = classify(name)
        groups[g]["params"] += n
        groups[g]["bytes"] += b
        groups[g]["tensors"] += 1
    return dict(groups)


def classify(name):
    if "weight_scale_inv" in name:
        return "fp8_scales"
    if name.startswith("model.visual") or ".visual." in name:
        return "vision_tower"
    if "embed_tokens" in name:
        return "embeddings"
    if name == "lm_head.weight" or "shared_head.head" in name:
        return "lm_head"
    if name == "model.norm.weight" or name.endswith("model.norm.weight"):
        return "final_norm"
    m = re.search(r"layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    if any(k in name for k in ("enorm", "hnorm", "eh_proj", "shared_head.norm")):
        return "mtp_glue"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".mlp.shared_experts." in name:
        return "shared_experts"
    if ".mlp.gate." in name or name.endswith(".mlp.gate.weight"):
        return "moe_router"
    if ".mlp." in name:
        return "dense_mlp"
    if ".self_attn.indexer" in name or "indexers_proj" in name:
        return "dsa_indexer"
    if ".self_attn." in name:
        return "attention"
    if ".linear_attn." in name:
        return "linear_attention"
    if "hyper" in name or re.search(r"\bhc\b|hc_", name) or "residual" in name:
        return "hyper_connections"
    if "layernorm" in name or name.endswith("norm.weight") or ".norm" in name:
        return "norms"
    return "other:" + re.sub(r"layers\.\d+\.", "layers.N.", name)


def layer_param_count(tensors, layer_ids, model_prefix=""):
    """Params per layer index."""
    per_layer = defaultdict(int)
    for name, info in tensors.items():
        if "weight_scale_inv" in name:
            continue
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            per_layer[int(m.group(1))] += numel(info["shape"])
    return dict(per_layer)


def kv_cache_math(tag, cfg, n_dsa_layers, n_indexer_full, kda_layers=0,
                  kda_heads=0, kda_head_dim=0, conv_kernel=0, mtp_layers=1,
                  rope_dim=None, kpool=1):
    """Per-token KV cache in elements, plus constant per-sequence state."""
    kv_lora = cfg["kv_lora_rank"]
    rope = cfg["qk_rope_head_dim"] if rope_dim is None else rope_dim
    latent_per_tok = n_dsa_layers * (kv_lora + rope)
    idx_per_tok = n_indexer_full * cfg["index_head_dim"] / kpool
    # KDA recurrent state: per sequence, not per token
    kda_state = kda_layers * kda_heads * kda_head_dim * kda_head_dim
    kda_conv = kda_layers * conv_kernel * kda_heads * kda_head_dim * 3  # q,k,v conv streams (approx)
    return {
        "model": tag,
        "dsa_layers_cached": n_dsa_layers,
        "latent_elems_per_token": latent_per_tok,
        "indexer_full_layers": n_indexer_full,
        "indexer_elems_per_token": idx_per_tok,
        "total_elems_per_token": latent_per_tok + idx_per_tok,
        "kb_per_token_bf16_latent_fp8_index": (latent_per_tok * 2 + idx_per_tok * 1) / 1000,
        "kb_per_token_all_fp8": (latent_per_tok + idx_per_tok) / 1000,
        "kda_state_elems_per_seq": kda_state,
        "kda_state_mb_bf16": kda_state * 2 / 1e6,
    }


def main():
    big_cfg = load("models/glm53_config.json")
    flash_cfg_all = load("models/glm53_flash_config.json")
    flash_cfg = flash_cfg_all["text_config"]
    big_t = load("results/glm53_tensors.json")
    flash_t = load("results/glm53_flash_tensors.json")
    big_idx = load("models/glm53_index.json")
    flash_idx = load("models/glm53_flash_index.json")

    out = {}

    for tag, tensors, idx in (("glm53", big_t, big_idx), ("glm53_flash", flash_t, flash_idx)):
        groups = group_tensors(tensors)
        total_params = sum(g["params"] for k, g in groups.items() if k != "fp8_scales")
        total_bytes = sum(g["bytes"] for g in groups.values())
        out[tag] = {
            "groups": groups,
            "total_params_excl_scales": total_params,
            "total_bytes_computed": total_bytes,
            "total_bytes_index_metadata": idx["metadata"]["total_size"],
            "n_tensors": len(tensors),
        }

    # sample tensor names for novel components
    def sample(tensors, pat, limit=40):
        hits = sorted(n for n in tensors if re.search(pat, n) and "scale_inv" not in n)
        return hits[:limit]

    out["samples"] = {
        "glm53_layer6_attn": sample(big_t, r"layers\.6\.self_attn\."),
        "glm53_layer7_attn": sample(big_t, r"layers\.7\.self_attn\."),
        "glm53_mtp78": sample(big_t, r"layers\.78\.(?!mlp\.experts)"),
        "flash_kda_layer0": sample(flash_t, r"layers\.0\.(?!mlp)"),
        "flash_dsa_layer3": sample(flash_t, r"layers\.3\.(?!mlp)"),
        "flash_hc": sample(flash_t, r"hc|hyper", 60),
        "flash_mtp45": sample(flash_t, r"layers\.45\.(?!mlp\.experts)"),
        "flash_vision": sample(flash_t, r"visual", 60),
    }

    # shapes of interest
    def shapes(tensors, names):
        return {n: tensors[n] for n in names if n in tensors}

    out["kv_cache"] = [
        kv_cache_math("GLM-5.3", big_cfg, n_dsa_layers=78, n_indexer_full=21),
        kv_cache_math("GLM-5.3-Flash", flash_cfg, n_dsa_layers=11, n_indexer_full=11,
                      kda_layers=34, kda_heads=64, kda_head_dim=128, conv_kernel=4,
                      rope_dim=0, kpool=flash_cfg.get("index_kpool", 1)),
        kv_cache_math("GLM-5.3-Flash (no kpool)", flash_cfg, n_dsa_layers=11, n_indexer_full=11,
                      kda_layers=34, kda_heads=64, kda_head_dim=128, conv_kernel=4,
                      rope_dim=0, kpool=1),
    ]

    # active parameter estimate per token (text path)
    def active_params(tag, cfg, tensors, dsa_layers, kda_layer_ids=None):
        per_expert = 3 * cfg["hidden_size"] * cfg["moe_intermediate_size"]
        n_layers = cfg["num_hidden_layers"]
        act = 0
        detail = {}
        # embeddings/lm_head not counted (lookup) except lm_head matmul
        act += cfg["vocab_size"] * cfg["hidden_size"]  # lm_head
        detail["lm_head"] = cfg["vocab_size"] * cfg["hidden_size"]
        groups = group_tensors(tensors)
        # attention + indexer + norms + dense mlp + routers + shared experts all active
        always = ["attention", "dsa_indexer", "linear_attention", "dense_mlp",
                  "moe_router", "shared_experts", "norms", "final_norm",
                  "hyper_connections"]
        for g in always:
            if g in groups:
                act += groups[g]["params"]
                detail[g] = groups[g]["params"]
        # routed experts: 8 of n per sparse layer (exclude MTP layer's experts? MTP idle in plain decode)
        n_sparse = cfg["num_hidden_layers"] - cfg["first_k_dense_replace"]
        routed_active = n_sparse * cfg["num_experts_per_tok"] * per_expert
        act += routed_active
        detail["routed_experts_active"] = routed_active
        # remove MTP-layer contributions counted in groups? MTP glue small; keep note.
        return act, detail

    a_big, d_big = active_params("glm53", big_cfg, big_t, 78)
    a_fl, d_fl = active_params("flash", flash_cfg,
                               {k: v for k, v in flash_t.items() if "visual" not in k}, 11)
    out["active_params"] = {"glm53": {"total": a_big, "detail": d_big},
                            "glm53_flash": {"total": a_fl, "detail": d_fl}}

    json.dump(out, open("results/architecture_analysis.json", "w"), indent=1)

    # ---- print summary ----
    for tag in ("glm53", "glm53_flash"):
        print("=" * 60)
        print(tag, f"  tensors={out[tag]['n_tensors']}")
        print(f"  total params (excl fp8 scales): {out[tag]['total_params_excl_scales']/1e9:.2f} B")
        print(f"  bytes computed {out[tag]['total_bytes_computed']/1e9:.1f} GB "
              f"vs index metadata {out[tag]['total_bytes_index_metadata']/1e9:.1f} GB")
        for k, v in sorted(out[tag]["groups"].items(), key=lambda x: -x[1]["params"]):
            print(f"    {k:20s} {v['params']/1e9:9.3f} B params  {v['tensors']:6d} tensors")
    print("=" * 60)
    print("ACTIVE (analytic):",
          f"glm53 {a_big/1e9:.1f}B", f"flash {a_fl/1e9:.1f}B")
    for row in out["kv_cache"]:
        print(row)


if __name__ == "__main__":
    main()

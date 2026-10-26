# glm53-teardown

Architecture teardown of **zai-org/GLM-5.3** (744B-A40B text model, DSA
attention) vs **zai-org/GLM-5.3-Flash** (320B-A18B natively multimodal
hybrid KDA + DSA model). No weights were downloaded and nothing was run
on the rig; this is a pure config + weight-map teardown.

## Raw sources of truth

- `models/glm53_config.json`: fetched 2026-08-30 from
  `zai-org/GLM-5.3` main branch.
- `models/glm53_flash_config.json`: fetched 2026-08-30 from
  `zai-org/GLM-5.3-Flash` main branch.
- `models/glm5_config.json`: `zai-org/GLM-5`, for lineage comparison.
- `models/glm53_index.json`, `models/glm53_flash_index.json`: the
  `model.safetensors.index.json` weight maps (tensor → shard, total_size).
- `results/glm53_tensors.json`, `results/glm53_flash_tensors.json`:
  complete `{tensor: {dtype, shape}}` maps built by
  `research/fetch_headers.py`, which reads the JSON header of every
  safetensors shard over HTTP range requests (141 shards for GLM-5.3,
  each header a few tens of KB; the weights themselves stay on the Hub).

## Analysis

- `research/analyze.py`: groups all tensors into components, computes
  parameter counts per component, derives KV-cache-per-token math, and
  writes `results/architecture_analysis.json`.
- `blog/make_figures.py`: renders the charts used by the draft.

All parameter counts and cache sizes here are **derived from the raw
files**, not measured on hardware. Benchmark scores quoted in the draft
are the vendor's self-reported numbers, marked as such.

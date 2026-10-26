#!/usr/bin/env bash
# Sweep Qwen 3.8 27B across MTP draft depths on macOS Metal backend
set -euo pipefail

MODEL_PATH="${1:-models/qwen3.8-27b-q4_k_m.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"

echo "Starting Apple Silicon Metal sweep for ${MODEL_PATH}..."
python3 benchmark_mac.py --server "${LLAMA_SERVER}" --model "${MODEL_PATH}" --repeats 3

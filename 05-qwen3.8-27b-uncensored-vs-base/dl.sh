#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="${1:-models}"
mkdir -p "$DEST_DIR"

BASE_URL="https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main"
FILE_NAME="Qwen3.8-27B-Uncensored-Q4_K_M.gguf"

AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
fi

echo "=== $(date +%H:%M:%S) Starting download: $FILE_NAME ==="
curl -L "${AUTH_HEADER[@]}" -C - -o "$DEST_DIR/$FILE_NAME" "$BASE_URL/$FILE_NAME"
echo "=== $(date +%H:%M:%S) Finished download: $FILE_NAME ==="

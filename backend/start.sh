#!/bin/sh
set -e
# If the index was not baked into the image (data is git-ignored), fetch it.
# Set DATA_URL to a tar.gz containing data/qdrant_remote + data/hnsw.
# Example: https://huggingface.co/datasets/<you>/konkan-data/resolve/main/data.tar.gz
if [ ! -f "data/qdrant_remote/chunks.jsonl" ] && [ -n "$DATA_URL" ]; then
  echo "[start] Index not found, fetching from $DATA_URL ..."
  mkdir -p data
  curl -L "$DATA_URL" -o /tmp/data.tar.gz
  tar -xzf /tmp/data.tar.gz -C .
  rm /tmp/data.tar.gz
  echo "[start] Index fetched."
elif [ ! -f "data/qdrant_remote/chunks.jsonl" ]; then
  echo "[start] WARNING: data/qdrant_remote/chunks.jsonl not found and DATA_URL not set."
  echo "         The API will start with 0 chunks (health will show 0)."
  echo "         Push data via Git LFS or set DATA_URL (see DEPLOY_RENDER.md)."
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

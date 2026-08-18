#!/usr/bin/env bash
# Qwen3.8-27B llama.cpp server launcher (CUDA)
# Set MODEL_PATH to your .gguf file before running if it moves.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-$HOME/Downloads/Qwen3.8-27B-Q5_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$HOME/claude-projects/llama.cpp/build/bin/llama-server}"
GPU_LAYERS="${GPU_LAYERS:-999}"
# 262144 (256K, this model's native context) with q8_0 KV cache uses ~28.75/32 GB
# VRAM on a 5090 (20.75GB Q5_K_M weights + ~8GB KV cache), leaving headroom for
# compute buffers. Single slot (--parallel 1) so the full budget goes to one
# session — Claude Code's system prompt + full tool schema alone can run 50-60K
# tokens, so don't drop this below ~65536.
CONTEXT_SIZE="${CONTEXT_SIZE:-262144}"
PORT="${PORT:-1234}"

# Sampling defaults follow the HF-recommended "Thinking Mode" profile (model card,
# Aug 2026) — bridge.py overrides these per-request to switch to the "Instruct
# Mode" profile when a request runs with thinking disabled; these are the fallback
# if llama-server is queried directly without the bridge in front of it.
exec "$LLAMA_SERVER" \
  -m "$MODEL_PATH" \
  -ngl "$GPU_LAYERS" \
  -c "$CONTEXT_SIZE" \
  -ctk q8_0 -ctv q8_0 -fa on \
  --load-mode none \
  --jinja \
  -b 2048 -ub 2048 \
  -t 16 --parallel 1 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --port "$PORT" \
  --host 127.0.0.1

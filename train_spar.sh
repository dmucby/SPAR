#!/usr/bin/env bash
set -euo pipefail

export WANDB_MODE="${WANDB_MODE:-offline}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

TORCHRUN_BIN="${TORCHRUN_BIN:-}"
if [ -z "$TORCHRUN_BIN" ]; then
    TORCHRUN_BIN="$(command -v torchrun || true)"
fi
if [ -z "$TORCHRUN_BIN" ]; then
    echo "torchrun not found. Set TORCHRUN_BIN or activate the SPAR environment." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
"$TORCHRUN_BIN" \
    --nproc_per_node "${NPROC_PER_NODE:-8}" \
    --nnodes 1 \
    --rdzv_id "${RDZV_ID:-18635}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${RDZV_ENDPOINT:-localhost:29504}" \
    train_spar.py \
    --config configs/spar/spar.yaml

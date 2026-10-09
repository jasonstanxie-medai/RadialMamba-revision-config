#!/usr/bin/env bash
# Published inference entry (nnU-Net predict_entry_point). Original oral predict_frozen.sh not on this Mac.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:?tag}"
TRAINER="${2:?trainer class}"
CKPT="${3:?checkpoint path}"
GPU="${4:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export RADIALMAMBA_EPOCH_LOCK="${RADIALMAMBA_EPOCH_LOCK:-$ROOT/configs/epoch_budget.json}"
exec nnUNetv2_predict -d Dataset893_IVOCT_holdout90 -c 2d -f 0 -tr "$TRAINER" -chk "$CKPT" -o "predictions_${TAG}" "$@"

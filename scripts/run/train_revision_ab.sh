#!/usr/bin/env bash
# Published wrapper derived from training_environment.md (oral launch_one.sh equivalent).
# Requires: PYTHONPATH to a full nnU-Net/Swin-UMamba tree that includes this repo's nnunetv2 overlays.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARM="${1:?usage: $0 A|B [extra nnUNet args]}"
export RADIALMAMBA_EPOCH_LOCK="${RADIALMAMBA_EPOCH_LOCK:-$ROOT/configs/epoch_budget.json}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
unset IVOCT_OVERSAMPLE_VV IVOCT_FOCAL_CLASS_WEIGHTS nnUNet_compile || true
case "$ARM" in
  A|a) TR=nnUNetTrainerSwinUMambaRevisionABCoord ;;
  B|b) TR=nnUNetTrainerSwinUMambaRevisionABNoCoord ;;
  *) echo "ARM must be A or B"; exit 1 ;;
esac
shift || true
exec python -m nnunetv2.run.run_training \
  Dataset893_IVOCT_holdout90 2d 0 \
  -tr "$TR" "$@"

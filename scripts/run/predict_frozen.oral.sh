#!/usr/bin/env bash
# Isolated imagesTs predict. CUDA_VISIBLE_DEVICES must be one of 3,4,5,6.
set -euo pipefail
TAG="${1:?tag e.g. A_main}"
TRAINER="${2:?trainer class}"
CKPT="${3:?checkpoint pth}"
GPU="${4:?3|4|5|6}"
REV="/root/IVOCT_RadialMamba/revision_ab"
UMAMBA="/root/IVOCT_RadialMamba/IVOCT_REMOTE_READY/U-Mamba-main/umamba"
PY="${REV}/.venv/bin/python"
SRC_PLANS="${REV}/nnUNet_results/Dataset893_IVOCT_holdout90/${TRAINER}__nnUNetPlans__2d"
DST_ROOT="${REV}/nnUNet_results_frozen/${TAG}"
DST="${DST_ROOT}/Dataset893_IVOCT_holdout90/${TRAINER}__nnUNetPlans__2d"
OUT="${REV}/predictions/${TAG}/imagesTs"
mkdir -p "${DST}/fold_0" "${OUT}"
cp -a "${SRC_PLANS}/plans.json" "${SRC_PLANS}/dataset.json" "${SRC_PLANS}/dataset_fingerprint.json" "${DST}/"
cp -a "${CKPT}" "${DST}/fold_0/checkpoint_final.pth"
cat > "${REV}/predictions/${TAG}/RUN_MANIFEST.json" <<EOF
{
  "tag": "${TAG}",
  "trainer": "${TRAINER}",
  "selected_checkpoint_src": "${CKPT}",
  "loaded_as": "checkpoint_final.pth",
  "split": "imagesTs",
  "n_images_expected": 1899,
  "save_probabilities": false,
  "coord": "AddCoordinates on the network input tensor (nnU-Net patch), not full-image catheter origin"
}
EOF
export PYTHONPATH="${UMAMBA}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export nnUNet_raw="${REV}/nnUNet_raw"
export nnUNet_preprocessed="${REV}/nnUNet_preprocessed"
export nnUNet_results="${DST_ROOT}"
export OMP_NUM_THREADS=1
unset nnUNet_compile
exec "${PY}" -c 'import sys; from nnunetv2.inference.predict_from_raw_data import predict_entry_point; sys.argv=["predict","-i",sys.argv[1],"-o",sys.argv[2],"-d","Dataset893_IVOCT_holdout90","-c","2d","-f","0","-tr",sys.argv[3],"-chk","checkpoint_final.pth"]; predict_entry_point()' \
  "${REV}/nnUNet_raw/Dataset893_IVOCT_holdout90/imagesTs" "${OUT}" "${TRAINER}"

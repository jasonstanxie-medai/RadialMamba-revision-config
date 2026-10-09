# Known runtime gaps

This repository publishes method code extracts, configs, splits, evaluation records, and analysis scripts. It does **not** redistribute model weights, raw IV-OCT images, or full per-pixel probability archives. Checkpoint SHA256 digests are listed in [`../reproducibility/checkpoints.json`](../reproducibility/checkpoints.json).

## Included

- Coordinate module and Revision A/B trainers under `src/` and `nnunetv2/`
- Train/predict wrappers under `scripts/run/`
- Study-level splits, primary recipes, and recorded metrics
- Unmodified upstream Swin-UMamba reference under `upstream/swin_umamba_stock/` (for attribution only)

## Not included (requires the archived training tree)

The training machine used a modified nnU-Net / Swin-UMamba source tree. The following components from that tree are not bundled here:

1. Training-modified `nnunetv2/nets/SwinUMamba.py` (coordinate factory arguments and forward-path `AddCoordinates`)
2. Parent trainers `nnUNetTrainerSwinUMamba.py` and `nnUNetTrainerSwinUMambaCoord.py` as used during training
3. `data_loader_2d_ivoct_vv.py` (used when VV oversampling is enabled in secondary recipes)
4. Full `compound_losses.py` containing `DC_and_Focal_loss` (this repo ships an excerpt only)
5. Complete `pip freeze` dump (recorded package versions appear in [`environment.md`](environment.md))

Do not substitute a fresh upstream checkout for item 1 without verifying that it accepts the same factory arguments as the Revision A/B trainers.

## Weights

Selected checkpoints remain offline. Contact the corresponding author if review access to weights is required.
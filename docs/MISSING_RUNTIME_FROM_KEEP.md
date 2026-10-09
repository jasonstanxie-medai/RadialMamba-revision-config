# Runtime files still needed from KEEP / training tree

**Status (2026-10-09, this Mac):** `/Volumes/My Passport/IVOCT_KEEP` is **not mounted**, and the oral path `/root/IVOCT_RadialMamba/...` is unreachable. No local byte-identical copy of the **training-modified** `SwinUMamba.py` was found.

Do **not** treat `upstream/swin_umamba_stock/` as the revision backbone: that tree is the **unmodified** Swin-UMamba release (Apache-2.0), kept only for license/reference.

## Recovered on this machine (committed)

| File | Source |
| --- | --- |
| `nnunetv2/nets/coord_conv.py` | Cursor history of training-era `coord_conv.py` |
| `nnunetv2/nets/UMambaEnc_2d.py` | Cursor history (coordinate-capable UMambaEnc; related path, not primary A/B) |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncCoord.py` | Cursor history |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainerUMambaEncCoordNoAMP.py` | Cursor history |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainerSwinUMambaRevisionAB.py` | Verified revision extract |
| `nnunetv2/training/nnUNetTrainer/nnUNetTrainerIVOCTRareLesion.py` | Verified revision extract |
| `nnunetv2/training/loss/softmax_focal_loss.py` | Verified revision extract |
| `scripts/run/train_revision_ab.sh` | Published wrapper from `training_environment.md` |
| `scripts/run/predict_frozen.sh` | Published nnU-Net predict wrapper (not oral script bytes) |
| `upstream/swin_umamba_stock/...` | Unmodified third_party Swin-UMamba |

## Still missing (copy from KEEP when disk returns)

Expected under oral/KEEP, e.g.  
`/root/IVOCT_RadialMamba/IVOCT_REMOTE_READY/U-Mamba-main/umamba/` or  
`/Volumes/My Passport/IVOCT_KEEP/handoff_20261007/` / full code tree:

1. `nnunetv2/nets/SwinUMamba.py` with `use_add_coordinates`, `add_coordinates_with_r`, `coord_init`, `coord_stem_baseline_ckpt`, `vmamba_ckpt_path`
2. `nnunetv2/training/nnUNetTrainer/nnUNetTrainerSwinUMamba.py` (parent used in training)
3. `nnunetv2/training/nnUNetTrainer/nnUNetTrainerSwinUMambaCoord.py`
4. `nnunetv2/training/dataloading/data_loader_2d_ivoct_vv.py` (+ deps)
5. Full `compound_losses.py` containing `DC_and_Focal_loss` (do not overwrite other loss classes with the excerpt)
6. Original `launch_one.sh`, oral `predict_frozen.sh`, `eval_locked.py`, `score_inner_val.py` if present
7. `audit/pip_freeze_revision_ab_venv.txt`
8. Dataset893 `dataset.json` and any label-conversion scripts used for nnU-Net packing

## Remount recipe for Cursor

```bash
# after My Passport is mounted
KEEP="/Volumes/My Passport/IVOCT_KEEP/handoff_20261007"
# or full umamba tree on Passport / oral rsync target
find "$KEEP" /Volumes/My\ Passport -name 'SwinUMamba.py' 2>/dev/null
# copy only verified training paths into nnunetv2/, then re-run:
python3 scripts/verify_bundle.py
```

Then update this file and the README **Snapshot scope** sentence.
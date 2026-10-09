# Runtime files: KEEP status (2026-10-09 remount)

**Weights / images / full `.npz` probability volumes are intentionally not published.** That matches common practice: release code, configs, splits, and checkpoint SHA256; share weights on request or via a separate controlled channel if needed.

## Present on My Passport KEEP and now in this repo

| Item | KEEP path → repo |
| --- | --- |
| Oral `predict_frozen.sh` | `handoff_…/scripts/predict_frozen.sh` → `scripts/run/predict_frozen.oral.sh` |
| `eval_locked.py` | → `scripts/run/eval_locked.py` |
| `score_inner_val.py` | → `scripts/run/score_inner_val.py` |
| Coord + RevisionAB extracts | already in `src/` / `nnunetv2/` |
| `dataset.json` (label map) | → `configs/dataset893/dataset.json` |
| Checkpoint SHA256 | `reproducibility/checkpoints.json` (files stay on KEEP only) |

## Still not on KEEP (never copied from oral full tree)

KEEP `code/` only stored extracts. The full modified tree  
`/root/IVOCT_RadialMamba/IVOCT_REMOTE_READY/U-Mamba-main/umamba/` was **not** in the handoff package.

Still missing unless recovered from oral GPU disk:

1. Training-modified `nnunetv2/nets/SwinUMamba.py` (`use_add_coordinates`, …)
2. `nnUNetTrainerSwinUMamba.py` / `nnUNetTrainerSwinUMambaCoord.py` as used in training
3. `data_loader_2d_ivoct_vv.py`
4. Full `compound_losses.py` containing `DC_and_Focal_loss` (repo has excerpt only)
5. `pip_freeze_revision_ab_venv.txt` (versions remain in `docs/environment.md` / legacy notes)
6. Original `launch_one.sh` (published wrapper: `scripts/run/train_revision_ab.sh`)

`upstream/swin_umamba_stock/` remains the **unmodified** upstream reference, not the revision backbone.

## Weights (local only — do not git add)

Example KEEP locations (multi‑GB; excluded by `.gitignore` intent):

- `archive/v1_locked_recipe_20261006_1110/A/checkpoint_latest.pth` (A@694)
- `backup/B/checkpoint_final.pth` (B@932)

Publish SHA256 only; optional later: Zenodo/Figshare private link or “available from corresponding author.”

# Training and inference

## Export status

The repository provides coordinate code, Revision A/B trainers, split/recipe records, evaluation artifacts, and run wrappers under `scripts/run/`. Files under `nnunetv2/` include `coord_conv.py` and related trainers.

It is **not** a self-contained clone-and-train install of the full modified backbone tree. See [RUNTIME_GAPS.md](RUNTIME_GAPS.md). Stock upstream Swin-UMamba under `upstream/swin_umamba_stock/` is an unmodified reference copy only.

The standalone split/result verification and coordinate example can run using the bundled files. Full training and checkpoint inference additionally require the archived modules below.

| Archived file or component | Why it is required |
| --- | --- |
| Modified `nnunetv2/nets/SwinUMamba.py` | Defines the coordinate-enabled model factory and forward pass |
| `nnUNetTrainerSwinUMamba.py` | Parent trainer, optimizer, freezing, and supervision scales |
| `nnUNetTrainerSwinUMambaCoord.py` | Imported by the rare-lesion trainer module |
| `nnUNetTrainerUMambaEncCoord.py` | Imported by the rare-lesion trainer module |
| `data_loader_2d_ivoct_vv.py` | Imported VV loader and the refinement recipe |
| Complete loss module or a documented patch | The included `compound_losses_excerpt_DC_and_Focal.py` is an excerpt, not a whole-file replacement |
| Dataset construction / `dataset.json` / fingerprint | Preserve RGB preparation and label conversion |
| Training and frozen-inference launch scripts | Preserve the source layout, checkpoint selection, and prediction settings |
| Original study scoring / bootstrap script | Re-run saved evaluation with the recorded aggregation and empty-mask rules |
| Complete environment export | Record all Mamba/SS2D and framework dependency versions |

Restore these files from the training backup. A fresh upstream checkout alone does not establish that it matches the archived modified model or accepts the added factory arguments.

## Integration of the supplied extracts

The trainer imports nnU-Net package paths. Copying the supplied Python files into an unrelated directory is insufficient. Integrate them into the archived source tree at their original package locations, or provide an explicit source patch against that tree. Preserve the original tree's notices and dependency setup.

The public trainer copy accepts two path variables:

- `RADIALMAMBA_EPOCH_LOCK`: path to the run-budget JSON. Its default points to the repository's `configs/epoch_budget.json`.
- `RADIALMAMBA_VMAMBA_CKPT`: path to the generic `vmamba_tiny_e292.pth` encoder checkpoint.

These environment-variable defaults change file lookup only, not the recorded loss, sampling, optimizer, or epoch budget. See [export notes](../reproducibility/export_notes.md).

## Recorded training entry point

Once the archived modified source tree, data, and pretrained encoder are installed, the recorded Python entry point is:

```bash
python -m nnunetv2.run.run_training Dataset893_IVOCT_holdout90 2d 0 \
  -tr nnUNetTrainerSwinUMambaRevisionABCoord
```

For B, replace the trainer with `nnUNetTrainerSwinUMambaRevisionABNoCoord`. The training run used `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`, a `PYTHONPATH` pointing to the archived tree, and one visible GPU per run. Place the supplied split in the dataset's preprocessed directory. Refer to the original launch scripts when restoring the environment.

This command records the entry point. End-to-end training has not been validated from this reduced export.

## Frozen checkpoint inference

The primary weights are A at epoch 694 and B at epoch 932. [`../reproducibility/checkpoints.json`](../reproducibility/checkpoints.json) records their SHA256 digests. Checkpoint files are not bundled.

Restore the archived prediction script and run each selected checkpoint with its matching trainer and plans. For probability metrics, save continuous per-class probability maps in `.npz` files using key `probabilities`. The score maps must align with the restored ground-truth image grid. A rerun of probability inference does not require retraining.

## Source projects

- [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba)
- [U-Mamba](https://github.com/bowang-lab/U-Mamba)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

These links acknowledge the upstream projects. Their present installation instructions are not a substitute for the recorded modified source snapshot.

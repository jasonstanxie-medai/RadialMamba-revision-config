# Export notes

Date: 2026-10-09.

This layout presents the revision experiment as a research code and verification package.

## Code changes in this public copy

1. `coord_conv.py`: correct the radius-range docstring to match the existing division by `sqrt(2)`. The numerical coordinate and initialization functions are unchanged.
2. `nnUNetTrainerSwinUMambaRevisionAB.py`: accept `RADIALMAMBA_EPOCH_LOCK` and `RADIALMAMBA_VMAMBA_CKPT` for local file locations. Default run-budget lookup uses the repository `configs/` directory. Correct the comment about main-process reseeding so it does not claim worker random sequences were aligned.
3. Numerical primary trainer settings are unchanged. No new training run, checkpoint, or result was created by these edits.

## Record and presentation

- The public study manifest reports the verified comma-joined validation-ID fingerprint.
- The English environment guide uses the training-machine export dated 2026-10-07.
- Architecture assets retain the manuscript illustration, with patch-local coordinates, unchanged SS2D, and recorded deep-supervision weights. The displayed medical case is an illustrative development example.
- Summary CSVs, corrected result JSONs, checkpoint hashes, the full split file, and the exact ROC/AP scoring script are unchanged numerical records.
- `scripts/verify_bundle.py` and `examples/coordinate_channels.py` are repository utilities for checking the published bundle; they are not the original training or bootstrap scripts.

## Validation scope

Split membership, fingerprints, recorded Dice arithmetic, documentation links, and Python syntax are checked in this package. These checks do not validate full model training from the reduced source export. Remaining modules are listed in [`../docs/RUNTIME_GAPS.md`](../docs/RUNTIME_GAPS.md).
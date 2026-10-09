# Export notes

Date: 2026-10-09. Source: the received `RadialMamba_revision_config_for_review.zip` plus the completed manuscript evaluation artifacts.

This layout presents the work as a research project. The original review ZIP remains preserved separately in the author's preparation package.

## Code changes in this public copy

1. `coord_conv.py`: correct the radius-range docstring to match the existing division by sqrt(2). Simplify the matching comment. The numerical coordinate and initialization functions are unchanged.
2. `nnUNetTrainerSwinUMambaRevisionAB.py`: accept `RADIALMAMBA_EPOCH_LOCK` and `RADIALMAMBA_VMAMBA_CKPT` for local file locations. Default run-budget lookup uses the sibling repository `configs/` directory. Correct the comment about main-process reseeding so it does not claim worker random sequences were aligned.
3. Numerical primary trainer settings are unchanged. No new training run, checkpoint, or result was created by these edits.

## Record and presentation changes

- The public study manifest gives the verified comma-joined validation-ID fingerprint and removes the older unverified-fingerprint note.
- The English environment guide uses the later training-machine export. The earlier configuration note's environment/launch uncertainty is not repeated as current status.
- The architecture export retains the author-provided illustration, clarifies patch-local coordinates and unchanged SS2D, labels height/width according to the code, and records the actual deep-supervision weights. The displayed medical case remains an illustrative development case.
- Summary CSVs, corrected result JSONs, checkpoint hashes, the full split file, and the exact ROC/AP scoring script are copied unchanged.
- The bundle checker and coordinate example are publication-time utilities. They are not the original training or bootstrap scripts.

## Validation scope

Split membership, fingerprints, recorded Dice arithmetic, local documentation links, and Python syntax are checked during preparation. The coordinate example requires PyTorch and was not executed in the preparation environment, which has no PyTorch installation. These checks do not validate full model training from the reduced source export. The remaining archived modules are listed in the [run guide](../docs/training_and_inference.md).

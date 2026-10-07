# RadialMamba revision config (private)

Private review-access package for Frontiers manuscript **1938608** (RadialMamba revision).

Contains verified training/eval configuration extracts, study-level split manifests, checkpoint SHA256 hashes, and selected source files. **No model weights and no raw IV-OCT images.**

## Contents

- `configuration_verified.md` — training and evaluation configuration extracted from saved plans, logs, and checkpoints
- `training_environment.md` — Python/PyTorch/CUDA versions and training launch commands from the training machine
- `locks/` — splits, epoch budget, eval rules, plans
- `code/` — coord conv, Revision A/B trainer, Dice+Focal loss excerpts
- `split_manifests/` — 74/19/10 study lists and checkpoint hashes
- `audit_refs/` — freeze records and main imagesTs metric refs

## Dataset (public, separate)

Source IV-OCT annotations: [Zenodo 10.5281/zenodo.14478210](https://doi.org/10.5281/zenodo.14478210).  
This repository does **not** redistribute the image data.

## Access

This repository is **private**. Collaborators / reviewers can be invited under GitHub → Settings → Collaborators.

# Upstream snapshots (not the revision-modified backbone)

`swin_umamba_stock/` is the **unmodified** Swin-UMamba / nnU-Net trainer extract from the local third_party mirror, retained for Apache-2.0 attribution and structural reference.

The revision A/B experiment used a **modified** `SwinUMamba.py` (coordinate kwargs + `AddCoordinates` in `forward`). That file is listed among the gaps in [`docs/RUNTIME_GAPS.md`](../docs/RUNTIME_GAPS.md).
# Available research source

| File | Role |
| --- | --- |
| `coord_conv.py` | Local x/y/r channels and input-weight inflation helpers |
| `nnUNetTrainerSwinUMambaRevisionAB.py` | Shared primary trainer settings and coordinate/no-coordinate arms |
| `nnUNetTrainerIVOCTRareLesion.py` | Dice–Focal/deep-supervision and optional rare-lesion sampling integration |
| `softmax_focal_loss.py` | Softmax focal loss |
| `compound_losses_excerpt_DC_and_Focal.py` | Compound loss extract, not a full replacement of the upstream module |

The coordinate example imports `coord_conv.py` directly. The trainer modules require their original nnU-Net package paths and additional archived source files. See [training and inference](../docs/training_and_inference.md).

Path and comment edits for local installation are listed in [export notes](../reproducibility/export_notes.md). These files do not invent a new backbone implementation.

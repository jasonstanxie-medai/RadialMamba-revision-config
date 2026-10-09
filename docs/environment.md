# Recorded training environment

These values come from the training-machine environment export dated 2026-10-07. They describe the completed experiments, rather than a tested installation specification for every new machine.

| Component | Recorded value |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.14.1+cu130 |
| CUDA reported by PyTorch | 13.0 |
| cuDNN integer | 92400 |
| NVIDIA driver | 595.91.07 |
| GPU | NVIDIA RTX 6000D, 80 GB |
| NumPy | 2.5.3 |
| batchgenerators | 0.25.3 |
| timm | 1.0.22 |
| MONAI | 1.6.1 |
| Pillow | 12.3.0 |
| nnU-Net | Modified source tree loaded through `PYTHONPATH`, not a pip-versioned package |

A single training Git HEAD was not retained for the unpacked/copied source tree. The source snapshot, manifests, and file hashes identify the exported materials. A future repository commit identifies the publication of these files, not a recovered training commit.

The archived full `pip freeze` is referenced in the handoff but is not included in the received configuration ZIP. It remains one of the source files to add to the public export.

## Analysis dependencies

[`../requirements-analysis.txt`](../requirements-analysis.txt) covers the probability evaluation script only. It does not install Swin-UMamba, the custom nnU-Net source tree, Mamba/SS2D extensions, or a compatible training environment. Obtain and verify those dependencies with the archived training tree before running checkpoints.

## Randomness

The primary run's global seed was not recorded. In the additional pair, seed 43 was set in the main process before initialization. Augmentation-worker randomness was not fully fixed, and cuDNN benchmarking was enabled. The additional pair uses final endpoints rather than repeating the primary checkpoint-selection policy. It is an endpoint sensitivity comparison, not a fully deterministic replication.

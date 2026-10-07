"""Multi-class focal loss on logits (nnU-Net layout)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SoftmaxFocalLoss(nn.Module):
    """
    Focal loss on softmax logits for dense multi-class segmentation.

    For each voxel with ground-truth class c:
        FL = alpha_c * (1 - p_c)^gamma * (-log p_c)

    alpha_c comes from optional per-class weights (Lin et al., ICCV 2017).
    Ignored voxels use ignore_index (default -100, nnU-Net convention).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Tensor | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        if class_weights is None:
            self.register_buffer("_alpha", torch.ones(1), persistent=False)
            self._use_alpha = False
        else:
            self.register_buffer("_alpha", class_weights.float().clone(), persistent=True)
            self._use_alpha = True

    def forward(self, net_output: Tensor, target: Tensor) -> Tensor:
        if target.ndim == net_output.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        t = target.long()
        ce = F.cross_entropy(net_output, t, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce.clamp(max=50.0))
        loss = (1.0 - pt).pow(self.gamma) * ce
        if self._use_alpha:
            alpha = self._alpha.to(loss.device, dtype=loss.dtype)[t]
            loss = loss * alpha
        if self.ignore_index >= 0:
            m = t != self.ignore_index
            return (loss * m.float()).sum() / m.float().sum().clamp(min=1.0)
        return loss.mean()

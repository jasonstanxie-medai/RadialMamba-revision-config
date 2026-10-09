"""
Coordinate channels for encoder input (CoordConv-style), optional radius r.

Used with UMambaEnc / ResidualMambaEncoder: concat (x, y) or (x, y, r) before stem.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


class AddCoordinates(nn.Module):
    """
    Append normalized spatial channels to a 2D feature map (B, C, H, W).

    x, y are in [-1, 1] over height and width. Optional r = sqrt(x^2 + y^2) in [0, sqrt(2)].
    """

    def __init__(self, with_r: bool = True):
        super().__init__()
        self.with_r = with_r

    @property
    def num_coord_channels(self) -> int:
        return 3 if self.with_r else 2

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size, _, x_dim, y_dim = input_tensor.size()
        device = input_tensor.device
        dtype = input_tensor.dtype

        denom_x = max(x_dim - 1, 1)
        denom_y = max(y_dim - 1, 1)

        xx_ones = torch.ones(1, y_dim, device=device, dtype=dtype)
        xx_range = torch.arange(x_dim, device=device, dtype=dtype).unsqueeze(1)
        xx_channel = torch.matmul(xx_range, xx_ones) / denom_x * 2 - 1

        yy_ones = torch.ones(x_dim, 1, device=device, dtype=dtype)
        yy_range = torch.arange(y_dim, device=device, dtype=dtype).unsqueeze(0)
        yy_channel = torch.matmul(yy_ones, yy_range) / denom_y * 2 - 1

        xx_channel = xx_channel.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, x_dim, y_dim)
        yy_channel = yy_channel.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, x_dim, y_dim)

        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr_channel = torch.sqrt(torch.clamp(xx_channel ** 2 + yy_channel ** 2, min=0.0))
            # Match x,y scale: r in [0,1] (was [0, sqrt(2)]); improves stem balance for coord slices
            rr_channel = rr_channel / (2.0**0.5)
            out = torch.cat([out, rr_channel], dim=1)

        return out


def inflate_stem_first_block_in_channels(
    model_state: Dict[str, Any],
    pretrained_state: Dict[str, Any],
    image_channels: int,
    new_in_channels: int,
    stem_block_idx: int = 0,
    coord_init: str = "zeros",
) -> Dict[str, Any]:
    """
    Build a new state dict from `pretrained_state` for a model whose stem first BasicResBlock
    accepts `new_in_channels` (image + coord), copying image-channel weights from UMambaEnc
    trained without coord channels.

    Patches encoder.stem.{stem_block_idx}.conv1 and .conv3 (1x1 shortcut) if present.

    Args:
        model_state: full state_dict of the target model (may be same object you mutate).
        pretrained_state: state_dict from checkpoint without coord channels.
        image_channels: number of image channels in the dataloader (e.g. 1 or 3).
        new_in_channels: stem first block input channels (image_channels + num_coord_channels).
        stem_block_idx: index of BasicResBlock inside encoder.stem (default 0).
        coord_init: "zeros" or "small_normal" for extra input slice of conv weights.
    """
    out = dict(model_state)
    prefix = f"encoder.stem.{stem_block_idx}."
    for name in ("conv1.weight", "conv1.bias", "conv3.weight", "conv3.bias"):
        key = prefix + name
        if key not in out or key not in pretrained_state:
            continue
        cur = out[key]
        old = pretrained_state[key]
        if name.endswith("weight") and cur.dim() == 4 and old.dim() == 4:
            o_out, o_in, k1, k2 = old.shape
            n_out, n_in, nk1, nk2 = cur.shape
            if n_in != new_in_channels or o_in != image_channels or o_out != n_out or k1 != nk1 or k2 != nk2:
                continue
            new_w = torch.zeros_like(cur)
            c_copy = min(image_channels, o_in, n_in)
            new_w[:, :c_copy] = old[:, :c_copy]
            if n_in > image_channels:
                extra = new_w[:, image_channels:]
                if coord_init == "small_normal":
                    nn.init.kaiming_normal_(extra, mode="fan_out", nonlinearity="leaky_relu")
                # else zeros (already)
            out[key] = new_w
        elif name.endswith("bias") and cur.shape == old.shape:
            out[key] = old.clone()
    return out


def load_pretrained_with_coord_stem(
    model: nn.Module,
    pretrained_path: str,
    image_channels: int,
    num_coord_channels: int,
    strict: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
) -> nn.IncompatibleKeys:
    """
    Load a checkpoint from a non-coord model into a coord-augmented model, inflating stem conv1/conv3.
    """
    try:
        ckpt = torch.load(pretrained_path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(pretrained_path, map_location=map_location)
    if isinstance(ckpt, dict) and "network_weights" in ckpt:
        pretrained_state = ckpt["network_weights"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        pretrained_state = ckpt["state_dict"]
    else:
        pretrained_state = ckpt
    new_in = image_channels + num_coord_channels
    merged = inflate_stem_first_block_in_channels(
        model.state_dict(),
        pretrained_state,
        image_channels=image_channels,
        new_in_channels=new_in,
    )
    return model.load_state_dict(merged, strict=strict)


def _checkpoint_to_state_dict(ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict) and "network_weights" in ckpt:
        return ckpt["network_weights"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def inflate_conv2d_in_channels(
    cur_weight: torch.Tensor,
    old_weight: Optional[torch.Tensor],
    image_channels: int,
    coord_init: str = "zeros",
) -> torch.Tensor:
    """Copy image-channel slices from old_weight; zero (or small) init for coord channels."""
    new_w = torch.zeros_like(cur_weight)
    if old_weight is not None and old_weight.dim() == cur_weight.dim():
        c_copy = min(image_channels, old_weight.shape[1], cur_weight.shape[1])
        if c_copy > 0:
            new_w[:, :c_copy] = old_weight[:, :c_copy]
    elif image_channels > 0:
        c_copy = min(image_channels, cur_weight.shape[1])
        if c_copy > 0:
            new_w[:, :c_copy] = cur_weight[:, :c_copy].clone()

    if cur_weight.shape[1] > image_channels:
        extra = new_w[:, image_channels:]
        if coord_init == "small_normal":
            nn.init.kaiming_normal_(extra, mode="fan_out", nonlinearity="relu")
            extra.mul_(0.1)
    return new_w


# SwinUMamba: stem + UNETR encoder1 take image+coord channels when AddCoordinates is enabled.
SWIN_UMAMBA_COORD_INPUT_WEIGHT_KEYS = (
    "stem.0.weight",
    "encoder1.layer.conv1.conv.weight",
    "encoder1.layer.conv3.conv.weight",
)


def inflate_swin_umamba_coord_input_layers(
    model: nn.Module,
    image_channels: int,
    baseline_state: Optional[Dict[str, Any]] = None,
    coord_init: str = "zeros",
    verbose: bool = True,
) -> None:
    """
    CoordConv warm-start for Swin-UMamba: copy RGB weights from a 3-ch baseline; coord slices = 0.
    """
    state = model.state_dict()
    for key in SWIN_UMAMBA_COORD_INPUT_WEIGHT_KEYS:
        if key not in state:
            continue
        cur = state[key]
        if cur.dim() != 4 or cur.shape[1] <= image_channels:
            continue
        old = baseline_state.get(key) if baseline_state is not None else None
        new_w = inflate_conv2d_in_channels(cur, old, image_channels, coord_init=coord_init)
        state[key] = new_w
        if verbose:
            img_std = new_w[:, :image_channels].std().item()
            coord_std = new_w[:, image_channels:].std().item()
            src = "baseline" if old is not None else "in-place"
            print(
                f"inflate_swin_umamba_coord: {key} <- {src} "
                f"img_std={img_std:.6f} coord_std={coord_std:.6f}",
                flush=True,
            )
    model.load_state_dict(state)


def load_checkpoint_state_dict(path: str, map_location: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location or "cpu")
    return _checkpoint_to_state_dict(ckpt)

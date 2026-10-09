"""
IVOCT: rare small-structure handling on top of standard nnU-Net training.

- Dice + softmax focal loss (Lin et al., ICCV 2017), optional per-class focal alphas.
- Optional stronger foreground patch oversampling (nnU-Net mechanism).

For literature-aligned alternatives without custom code, see also:
- nnUNetTrainerDiceTopk10Loss (Dice + top-k CE) in variants/loss/nnUNetTrainerTopkLoss.py
- Single-class VV head / ensemble (e.g. Danilov et al. Swin-UMamba discussion): separate task, not implemented here.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_Focal_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_2d_ivoct_vv import nnUNetDataLoader2D_IVOCT_VV
from typing import Tuple

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUMamba import nnUNetTrainerSwinUMamba
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUMambaCoord import nnUNetTrainerSwinUMambaCoord
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEncCoord import nnUNetTrainerUMambaEncCoord


def _parse_ivoct_focal_class_weights(num_classes: int) -> torch.Tensor | None:
    s = os.environ.get("IVOCT_FOCAL_CLASS_WEIGHTS", "").strip()
    if not s:
        return None
    parts = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(parts) != num_classes:
        print(
            f"IVOCT_FOCAL_CLASS_WEIGHTS: expected {num_classes} values (label order including background); "
            f"got {len(parts)}. Using unweighted focal."
        )
        return None
    return torch.tensor(parts, dtype=torch.float32)


class IVOCTVVOversampleMixin:
    """Use VV-biased 2D training patches when IVOCT_OVERSAMPLE_VV is set (e.g. 0.3)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oversample_vv_percent = 0.0
        if (v := os.environ.get("IVOCT_OVERSAMPLE_VV", "").strip()) != "":
            self.oversample_vv_percent = float(v)
            self.print_to_log_file(
                f"IVOCT_OVERSAMPLE_VV={self.oversample_vv_percent} "
                f"(force-fg patches biased to label {os.environ.get('IVOCT_VV_LABEL', '4')})",
                also_print_to_console=True,
            )

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        if self.oversample_vv_percent <= 0 or dim != 2:
            return super().get_plain_dataloaders(initial_patch_size, dim)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoader2D_IVOCT_VV(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            oversample_vv_percent=self.oversample_vv_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )
        dl_val = nnUNetDataLoader2D(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
        )
        return dl_tr, dl_val


def build_dc_focal_with_deep_supervision(trainer) -> nn.Module:
    assert not trainer.label_manager.has_regions, "DC_and_Focal_loss: regions not supported"
    focal_gamma = float(os.environ.get("IVOCT_FOCAL_GAMMA", "2.0"))
    weight_focal = float(os.environ.get("IVOCT_WEIGHT_FOCAL", "1.0"))
    weight_dice = float(os.environ.get("IVOCT_WEIGHT_DICE", "1.0"))
    cw = _parse_ivoct_focal_class_weights(trainer.label_manager.num_segmentation_heads)
    loss = DC_and_Focal_loss(
        {
            "batch_dice": trainer.configuration_manager.batch_dice,
            "smooth": 1e-5,
            "do_bg": False,
            "ddp": trainer.is_ddp,
        },
        focal_gamma=focal_gamma,
        focal_class_weights=cw,
        weight_focal=weight_focal,
        weight_dice=weight_dice,
        ignore_label=trainer.label_manager.ignore_label,
        dice_class=MemoryEfficientSoftDiceLoss,
    )
    if trainer.enable_deep_supervision:
        deep_supervision_scales = trainer._get_deep_supervision_scales()
        weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()
        loss = DeepSupervisionWrapper(loss, weights)
    return loss


class nnUNetTrainerUMambaEncCoordDiceFocal(nnUNetTrainerUMambaEncCoord):
    """
    UMambaEnc + Coord + Dice + Focal.

    Environment (optional):
      IVOCT_FOCAL_GAMMA — default 2.0
      IVOCT_WEIGHT_FOCAL, IVOCT_WEIGHT_DICE — default 1.0
      IVOCT_FOCAL_CLASS_WEIGHTS — comma-separated alphas, length = num classes (bg,1,2,3,4 for IVOCT); e.g. 1,1,1,1,4
      IVOCT_OVERSAMPLE_FG — if set, overrides foreground oversampling fraction (e.g. 0.5)
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if (v := os.environ.get("IVOCT_OVERSAMPLE_FG", "").strip()) != "":
            self.oversample_foreground_percent = float(v)
            self.print_to_log_file(
                f"IVOCT_OVERSAMPLE_FG={self.oversample_foreground_percent} (foreground patch oversampling)"
            )

    def _build_loss(self):
        self.print_to_log_file(
            "Loss: DC_and_Focal_loss (IVOCT). Tune IVOCT_FOCAL_* / IVOCT_OVERSAMPLE_FG for rare VV."
        )
        return build_dc_focal_with_deep_supervision(self)


class nnUNetTrainerSwinUMambaDiceFocal(IVOCTVVOversampleMixin, nnUNetTrainerSwinUMamba):
    """
    Swin-UMamba without CoordConv, with Dice + Focal (IVOCT).

    Ablates the loss / oversampling contribution separately from nnUNetTrainerSwinUMambaCoordDiceFocal.
    Same IVOCT_* env as nnUNetTrainerSwinUMambaCoordDiceFocal.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if (v := os.environ.get("IVOCT_OVERSAMPLE_FG", "").strip()) != "":
            self.oversample_foreground_percent = float(v)
            self.print_to_log_file(
                f"IVOCT_OVERSAMPLE_FG={self.oversample_foreground_percent} (foreground patch oversampling)"
            )

    def _build_loss(self):
        self.print_to_log_file(
            "Loss: DC_and_Focal_loss (IVOCT), no Coord. Tune IVOCT_FOCAL_* / IVOCT_OVERSAMPLE_FG for rare VV."
        )
        return build_dc_focal_with_deep_supervision(self)


class nnUNetTrainerSwinUMambaCoordDiceFocal(IVOCTVVOversampleMixin, nnUNetTrainerSwinUMambaCoord):
    """Swin-UMamba + Coord + Dice + Focal; same IVOCT_* env as nnUNetTrainerUMambaEncCoordDiceFocal."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if (v := os.environ.get("IVOCT_OVERSAMPLE_FG", "").strip()) != "":
            self.oversample_foreground_percent = float(v)
            self.print_to_log_file(
                f"IVOCT_OVERSAMPLE_FG={self.oversample_foreground_percent} (foreground patch oversampling)"
            )

    def _build_loss(self):
        self.print_to_log_file(
            "Loss: DC_and_Focal_loss (IVOCT). Tune IVOCT_FOCAL_* / IVOCT_OVERSAMPLE_FG for rare VV."
        )
        return build_dc_focal_with_deep_supervision(self)


class nnUNetTrainerTangExternalFinetune(nnUNetTrainerSwinUMambaCoordDiceFocal):
    """Tang external fine-tune with EMA early stop (avoids 1000-epoch overfit on 894)."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = int(os.environ.get("TANG_FINETUNE_MAX_EPOCHS", "300"))
        self.initial_lr = float(os.environ.get("TANG_FINETUNE_LR", "5e-5"))
        self.oversample_foreground_percent = float(os.environ.get("IVOCT_OVERSAMPLE_FG", "0.35"))
        self.early_stop_patience = int(os.environ.get("TANG_EARLY_STOP_PATIENCE", "40"))
        self.early_stop_min_epochs = int(os.environ.get("TANG_EARLY_STOP_MIN_EPOCHS", "60"))
        self.early_stop_min_delta = float(os.environ.get("TANG_EARLY_STOP_MIN_DELTA", "0.002"))
        self._early_stop_best_ema = -1.0
        self._early_stop_patience_counter = 0
        self._stop_training = False
        self.print_to_log_file(
            f"TangExternalFinetune: max_epochs={self.num_epochs} lr={self.initial_lr} "
            f"early_stop patience={self.early_stop_patience} min_epochs={self.early_stop_min_epochs}",
            also_print_to_console=True,
        )

    def on_epoch_end(self):
        epoch_done = self.current_epoch
        ema = (
            float(self.logger.my_fantastic_logging["ema_fg_dice"][-1])
            if self.logger.my_fantastic_logging["ema_fg_dice"]
            else 0.0
        )
        nnUNetTrainerSwinUMambaCoordDiceFocal.on_epoch_end(self)

        if epoch_done >= self.early_stop_min_epochs:
            if ema > self._early_stop_best_ema + self.early_stop_min_delta:
                self._early_stop_best_ema = ema
                self._early_stop_patience_counter = 0
            else:
                self._early_stop_patience_counter += 1
            if self._early_stop_patience_counter >= self.early_stop_patience:
                self._stop_training = True
                self.print_to_log_file(
                    f"Tang early stop at epoch {epoch_done}: best EMA fg dice={self._early_stop_best_ema:.4f}",
                    also_print_to_console=True,
                )

    def run_training(self):
        self.on_train_start()
        self._stop_training = False
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
            if self._stop_training:
                break
        self.on_train_end()

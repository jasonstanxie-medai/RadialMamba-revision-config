"""Revision A/B trainers. The two classes differ only in coordinate channels.

Shared recipe is fixed in code. Epoch count and batch size for the real run
come from revision_ab/locks/epoch_budget.json, written before training starts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from torch import nn

from nnunetv2.nets.SwinUMamba import get_swin_umamba_from_plans
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerIVOCTRareLesion import (
    nnUNetTrainerSwinUMambaDiceFocal,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager

LOCK_PATH = os.environ.get(
    "RADIALMAMBA_EPOCH_LOCK",
    str(Path(__file__).resolve().parents[1] / "configs" / "epoch_budget.json"),
)
VMAMBA_CKPT = os.environ.get(
    "RADIALMAMBA_VMAMBA_CKPT",
    "",  # set to local vmamba_tiny_e292.pth; required before training
)

# Identical for A and B. Do not read SWIN_COORD_* or old fold checkpoints.
SHARED = {
    "initial_lr": 1e-4,
    "weight_decay": 5e-2,
    "freeze_encoder_epochs": 10,
    "stem_lr_mult": 1.0,
    "warmup_epochs": 0,
    "oversample_foreground_percent": 0.5,
    "oversample_vv_percent": 0.0,
    "focal_gamma": 2.0,
    "weight_focal": 1.0,
    "weight_dice": 1.0,
    "deep_supervision": True,
    "coord_init": "zeros",
    "save_every": 1,
}


def _apply_revab_seed(torch_too: bool) -> None:
    """Seed 2 sets this. Seed 1 leaves it unset and keeps the nnU-Net default RNG."""
    raw = os.environ.get("REVAB_SEED", "").strip()
    if raw == "":
        return
    seed = int(raw)
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    if torch_too:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _apply_shared_env() -> None:
    os.environ["IVOCT_OVERSAMPLE_FG"] = str(SHARED["oversample_foreground_percent"])
    os.environ["IVOCT_FOCAL_GAMMA"] = str(SHARED["focal_gamma"])
    os.environ["IVOCT_WEIGHT_FOCAL"] = str(SHARED["weight_focal"])
    os.environ["IVOCT_WEIGHT_DICE"] = str(SHARED["weight_dice"])
    os.environ.pop("SWIN_COORD_LR", None)
    os.environ.pop("SWIN_COORD_STEM_LR_MULT", None)
    os.environ.pop("SWIN_COORD_FREEZE_EPOCHS", None)
    os.environ.pop("SWIN_COORD_USE_PRETRAIN", None)
    os.environ.pop("SWIN_COORD_STEM_FROM_BASELINE", None)
    os.environ["SWIN_VMAMBA_CKPT"] = VMAMBA_CKPT
    # Short finetune from latest: raise rare-class sampling/loss. A and B get the same values.
    if os.environ.get("REVAB_FT_EXTRA", "").strip():
        os.environ["IVOCT_OVERSAMPLE_VV"] = os.environ.get("REVAB_FT_VV_OS", "0.3")
        os.environ["IVOCT_FOCAL_CLASS_WEIGHTS"] = os.environ.get("REVAB_FT_FOCAL_W", "1,1,2,2,4")
    else:
        os.environ.pop("IVOCT_OVERSAMPLE_VV", None)
        os.environ.pop("IVOCT_FOCAL_CLASS_WEIGHTS", None)


def _epochs_and_batch() -> tuple[int, int]:
    if os.environ.get("REVAB_SPEEDTEST") == "1":
        return int(os.environ["REVAB_NUM_EPOCHS"]), int(os.environ["REVAB_BATCH_SIZE"])
    with open(LOCK_PATH) as f:
        lock = json.load(f)
    if lock.get("early_stop") not in (None, "null"):
        raise RuntimeError(f"epoch lock must not enable early stop: {lock.get('early_stop')}")
    return int(lock["num_epochs"]), int(lock["batch_size"])


def _build(use_coords: bool, plans_manager, dataset_json, configuration_manager, num_input_channels, enable_deep_supervision):
    if not os.path.isfile(VMAMBA_CKPT) or os.path.getsize(VMAMBA_CKPT) < 50_000_000:
        raise FileNotFoundError(f"generic VMamba checkpoint missing or too small: {VMAMBA_CKPT}")
    model = get_swin_umamba_from_plans(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        deep_supervision=enable_deep_supervision,
        use_pretrain=True,
        use_add_coordinates=use_coords,
        add_coordinates_with_r=True,
        vmamba_ckpt_path=VMAMBA_CKPT,
        coord_stem_baseline_ckpt=None,
        coord_init="zeros",
    )
    return model


class _RevisionAB(nnUNetTrainerSwinUMambaDiceFocal):
    """Shared Dice-Focal Swin-UMamba recipe. Subclasses set use_coords."""

    use_coords = False

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset: bool = True, device=None):
        import torch

        _apply_shared_env()
        epochs, batch = _epochs_and_batch()
        os.environ["SWIN_UMAMBA_BATCH_CAP"] = str(batch)
        if device is None:
            device = torch.device("cuda")
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        if self.fold != 0:
            raise RuntimeError("revision A/B uses only fold 0 (the study-level inner-val)")
        self.initial_lr = SHARED["initial_lr"]
        self.weight_decay = SHARED["weight_decay"]
        self.freeze_encoder_epochs = SHARED["freeze_encoder_epochs"]
        self.enable_deep_supervision = SHARED["deep_supervision"]
        self.num_epochs = epochs
        self.save_every = SHARED["save_every"]
        self.oversample_foreground_percent = SHARED["oversample_foreground_percent"]
        self.unpack_dataset = False
        if self.batch_size != batch:
            self.batch_size = batch
        if self.batch_size != batch or self.initial_lr != SHARED["initial_lr"]:
            raise RuntimeError(
                f"A/B lock failed: batch {self.batch_size} lr {self.initial_lr}"
            )
        if self.freeze_encoder_epochs != 10:
            raise RuntimeError("freeze_encoder_epochs drifted")
        arm = "A_coords" if self.use_coords else "B_nocoord"
        seed_note = os.environ.get("REVAB_SEED", "").strip() or "default"
        vv_os = os.environ.get("IVOCT_OVERSAMPLE_VV", "0")
        focal_w = os.environ.get("IVOCT_FOCAL_CLASS_WEIGHTS", "none")
        self.print_to_log_file(
            f"REVAB arm={arm} coords={int(self.use_coords)} in_ch_extra={3 if self.use_coords else 0} "
            f"lr={self.initial_lr} wd={self.weight_decay} freeze={self.freeze_encoder_epochs} "
            f"stem_lr_mult=1 warmup=0 batch={self.batch_size} epochs={self.num_epochs} "
            f"loss=DC_and_Focal fg_os=0.5 vv_os={vv_os} focal_w={focal_w} ds=1 opt=AdamW sched=CosineAnnealing "
            f"pretrain=vmamba_tiny_e292 coord_init=zeros baseline_ckpt=none early_stop=none "
            f"save_every=1 unpack=0 seed={seed_note}",
            also_print_to_console=True,
        )

    def initialize(self):
        _apply_revab_seed(torch_too=True)
        super().initialize()

    def on_train_start(self):
        # Reseed the main process. The archived augmentation workers use
        # seeds=None, so this does not align their random sequences.
        _apply_revab_seed(torch_too=False)
        super().on_train_start()
        extra = os.environ.get("REVAB_FT_EXTRA", "").strip()
        if extra == "":
            return
        extra_n = int(extra)
        if extra_n < 1:
            raise RuntimeError("REVAB_FT_EXTRA must be >= 1")
        from torch.optim.lr_scheduler import CosineAnnealingLR

        class _OffsetCosine(CosineAnnealingLR):
            def __init__(self, optimizer, t_max, eta_min, origin):
                self._origin = int(origin)
                super().__init__(optimizer, T_max=t_max, eta_min=eta_min)

            def step(self, epoch=None):
                if epoch is not None:
                    epoch = max(0, int(epoch) - self._origin)
                return super().step(epoch)

        self.num_epochs = self.current_epoch + extra_n
        ft_lr = float(os.environ.get("REVAB_FT_LR", "5e-5"))
        for group in self.optimizer.param_groups:
            group["lr"] = ft_lr
            group["initial_lr"] = ft_lr
        self.lr_scheduler = _OffsetCosine(
            self.optimizer, extra_n, 1e-5, self.current_epoch
        )
        self.print_to_log_file(
            f"REVAB_FT extra={extra_n} resume_epoch={self.current_epoch} "
            f"stop_epoch={self.num_epochs} lr={ft_lr} cosine eta_min=1e-5 offset "
            f"vv_os={os.environ.get('IVOCT_OVERSAMPLE_VV')} "
            f"focal_w={os.environ.get('IVOCT_FOCAL_CLASS_WEIGHTS')} "
            f"(A/B same; v1 dirs not used)",
            also_print_to_console=True,
        )

    def on_epoch_end(self):
        # Keep checkpoint_latest every epoch. Do not also write checkpoint_<n>.pth.
        nnUNetTrainer.on_epoch_end(self)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        raise NotImplementedError


class nnUNetTrainerSwinUMambaRevisionABCoord(_RevisionAB):
    use_coords = True

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _build(
            True,
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision,
        )


class nnUNetTrainerSwinUMambaRevisionABNoCoord(_RevisionAB):
    use_coords = False

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return _build(
            False,
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision,
        )

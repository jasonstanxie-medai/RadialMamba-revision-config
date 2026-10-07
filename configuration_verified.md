# configuration_verified.md

**用途：** 回应 `CURSOR_LOCAL_CONFIG_REQUEST.md`，为 GPT 正文 / 补充材料 / R1 第 6 条提供可核对字段。  
**原则：** 只写本地 KEEP 备份或可只读拉取的实现；无法核实写明「无法核实」；不猜 nnU-Net 默认值。  
**主证据根：** `/Volumes/My Passport/IVOCT_KEEP/handoff_20261007`（下称 KEEP）。  
**附小包：** `config_verified_bundle/`（本目录旁）。

---

## 1. Patch / plans / 空间维度

| 项 | 核实值 | 来源 |
|---|---|---|
| 配置名 | `2d` | KEEP `backup/A/plans.json`；A/B 训练日志 |
| patch | **512 × 512** | `configurations.2d.patch_size: [512, 512]` |
| 中位图尺寸 | 750 × 750（voxels） | `median_image_size_in_voxels`；`original_median_shape_after_transp: [1,750,750]` |
| spacing | `[1.0, 1.0]`（2d） | plans |
| 数据维 | 2D 自然图像（`NaturalImage2DIO`），通道 R/G/B | checkpoint `init_args.dataset_json` |
| batch（plans） | 12 | plans；锁定见 `locks/epoch_budget.json` |
| batch_dice | **True** | plans `batch_dice: true` |
| 数据集 | `Dataset893_IVOCT_holdout90` | plans / logs |

---

## 2. 坐标通道（`coord_conv.py`）与推理一致性

**实现（KEEP `code/coord_conv.py`）：**

- 在当前 2D 张量 `(B,C,H,W)` 上追加通道；**H→x 轴，W→y 轴**（代码：`x_dim,y_dim = H,W`）。
- \(x_{i}=2\cdot i/(H-1)-1\)，\(y_{j}=2\cdot j/(W-1)-1\)，范围 **[-1, 1]**（`H=1` 或 `W=1` 时分母取 1）。
- \(r=\sqrt{x^2+y^2}/\sqrt{2}\)，范围 **[0, 1]**（先 clamp≥0，再除 \(\sqrt{2}\)）。
- A 臂：`AddCoordinates(with_r=True)` → 图像 3 + 坐标 3；B 臂无坐标。
- 网络入口：`SwinUMamba.forward` 中 `x = self.add_coords(x_in) if ...`（autodl 上 `SwinUMamba.py` L604）；**在 crop / SpatialTransform / mirror 之后**、进入 stem 之前重建，故镜像后仍是「当前 patch 列 0 = −1」。
- 不是导管锁定全图径向先验。审计：`audit/coord_frame_check.json`（中位 750、patch 512、推理 tile 起点 `[0,238]`、约 4 tiles）。
- 推理：`predict_frozen.sh` 用同一 trainer 类；RUN_MANIFEST 写明同一 patch 内 AddCoordinates。

---

## 3. Dice / Focal / deep supervision

| 项 | 核实值 | 来源 |
|---|---|---|
| 损失 | `DC_and_Focal_loss` | 训练日志；`nnUNetTrainerIVOCTRareLesion.py` |
| Dice | `MemoryEfficientSoftDiceLoss`；**`do_bg=False`**（不含 background）；**`batch_dice=True`**；**`smooth=1e-5`** | `build_dc_focal_with_deep_supervision` |
| Focal | `SoftmaxFocalLoss`；γ 由 `IVOCT_FOCAL_GAMMA`（主配方 **2.0**）；**无 class alpha**（主配方 pop `IVOCT_FOCAL_CLASS_WEIGHTS`） | RareLesion + `RevisionAB` SHARED / env |
| Focal reduction | `F.cross_entropy(..., reduction="none")` 后按体素 `loss.mean()`（无 ignore 时）；有 ignore 则 mask 平均 | `softmax_focal_loss.py` |
| 权重 | `weight_dice=1`，`weight_focal=1` | SHARED + env |
| DS | 开启；SwinUMamba 父类尺度 **4 层**：`[[1,1],[0.5,0.5],[0.25,0.25],[0.125,0.125]]`；权重公式 `1/2^i`，**最后一层置 0 再归一化** → **[0.57142857, 0.28571429, 0.14285714, 0]** | `nnUNetTrainerSwinUMamba._get_deep_supervision_scales` + `build_dc_focal_with_deep_supervision` |

标签索引：`background=0, Lumen=1, Fibrous cap=2, Lipid core=3, Vasa vasorum=4`。

---

## 4. FG oversampling 0.5；VV sampling 关闭

- `SHARED["oversample_foreground_percent"]=0.5` → env `IVOCT_OVERSAMPLE_FG=0.5`。
- Loader：`base_data_loader._oversample_last_XX_percent`：batch 内下标  
  `j >= round(batch_size * (1-0.5))` → **后 50%（batch=12 时后 6 个）强制前景 bbox**。
- 主配方：`oversample_vv_percent=0.0`，且未设 `REVAB_FT_EXTRA` → **不进入** `nnUNetDataLoader2D_IVOCT_VV`。
- **日志证据（原文）：**  
  - A：`IVOCT_OVERSAMPLE_FG=0.5`；`REVAB arm=A_coords ... fg_os=0.5 vv_os=0 ...`  
  - B：`REVAB arm=B_nocoord ... fg_os=0.5 vv_os=0 ...`  
  文件：KEEP `backup/A|B/training_log_2026_10_5_12_08_03.txt`。

---

## 5. Preprocessing / augmentation / loader

**Preprocessing（plans `DefaultPreprocessor`）：**

- 每通道 **ZScoreNormalization**（3 通道）；`use_mask_for_norm=[False,False,False]`。
- 重采样：`resample_data_or_seg_to_shape`；data order 3，seg order 1。
- 前景强度统计见 plans `foreground_intensity_properties_per_channel`（0–255 自然图）。
- 随机裁剪在 dataloader（非 SpatialTransform `random_crop`）。

**Augmentation（nnU-Net 默认 `get_training_transforms`；2d、`do_dummy_2d_data_aug=False`，日志已确认）：**

| Transform | 幅度 / 概率 |
|---|---|
| SpatialTransform 旋转 | 方形 patch → **±180°**（绕 x）；`p_rot_per_sample=0.2` |
| SpatialTransform 缩放 | **(0.7, 1.4)**；`p_scale_per_sample=0.2`；无弹性形变 |
| GaussianNoise | `p=0.1` |
| GaussianBlur | σ (0.5,1.0)；`p_per_sample=0.2`，`p_per_channel=0.5` |
| BrightnessMultiplicative | (0.75,1.25)；`p=0.15` |
| ContrastAugmentation | `p=0.15` |
| SimulateLowResolution | zoom (0.5,1)；`p=0.25` |
| Gamma | (0.7,1.5) invert=True `p=0.1`；invert=False `p=0.3` |
| Mirror | axes **(0,1)** |

**Loader：** `num_iterations_per_epoch=250`，`num_val_iterations_per_epoch=50`（nnU-Net 基类默认；日志 epoch≈114–126 s 与测速一致）；`unpack_dataset=False`（RevisionAB）；fold **0 only**。

---

## 6. 优化器 / 日程 / 冻结 / batch·epochs

| 项 | 核实值 | 来源 |
|---|---|---|
| 优化器 | AdamW；lr **1e-4**；wd **0.05**；eps **1e-5**；betas **(0.9,0.999)** | 日志 Parameter Group + AB_CONFIG_DIFF |
| 日程 | `CosineAnnealingLR`；**T_max=num_epochs(932)**；**eta_min=1e-6**（SwinUMamba `configure_optimizers`）；无 warmup；无 restart | 父类 + AB_CONFIG_DIFF；日志末轮打印 `0.0` 为格式化近似 |
| 冻结 | encoder **前 10 epoch**（日志 Epoch 0–9 `Freezing`，Epoch 10+ `Unfreezing`） | 日志 + SHARED |
| batch / epochs | **12 / 932** | `locks/epoch_budget.json`；两边读同一锁 |
| iterations | **250 train / 50 val** per epoch | nnU-Net 基类（父类未改） |

---

## 7. 推理

| 项 | 核实值 | 来源 |
|---|---|---|
| 入口 | `nnunetv2.inference.predict_from_raw_data.predict_entry_point` | `predict_frozen.sh` |
| patch | 512×512（与 plans 一致） | plans / coord_frame_check |
| tile overlap | 默认 **`tile_step_size=0.5`**（步长约 256；750 图起点 0 与 238） | nnU-Net predictor 默认；`coord_frame_check.json` |
| Gaussian blending | **True**（默认） | predictor 默认；脚本未改 |
| TTA mirroring | 默认开启；checkpoint `inference_allowed_mirroring_axes=(0,1)`；脚本未传 `--disable_tta` | checkpoint + predict 默认 |
| 尺寸恢复 | nnU-Net 标准 resample 回原图 | predictor |
| 五类索引 | 0–4 见上 | dataset.json |
| 概率 vs 硬预测 | 主硬掩膜 `save_probabilities=false`；概率 ROC 另推 npz/softmax（`audit/roc_pr/`，tied exact 见 package evidence） | RUN_MANIFEST；handoff |

---

## 8. 环境 / 版本 / Git / 启动命令

| 项 | 状态 |
|---|---|
| oral 训练机 `revision_ab/.venv` 的 Python/PyTorch/nnU-Net/CUDA/cuDNN **精确版本** | **无法核实**（KEEP 无 `pip freeze`；`oral-gpu` SSH 公钥拒绝） |
| Git commit（训练代码树） | **无法核实**（KEEP 无 `.git`；oral 不可达） |
| 旁证（非 oral 训练环境，勿写入「训练机版本」） | autodl `umamba/.venv`：Python 3.12.3，torch 2.11.0+cu128，cudnn 91900（nnunetv2 需 PYTHONPATH） |
| 硬件（稿件已写） | RTX 6000D 80GB — KEEP 未单独存 GPU 查询日志；**不作新核实** |
| 启动（推理，已核实脚本） | `bash predict_frozen.sh <TAG> <TRAINER> <CKPT> <GPU>` |
| 启动（训练） | 训练 CLI 全文 **无法核实**（KEEP 无完整 `nnUNetv2_train` 命令行存档）；锁定参数见 trainer + `epoch_budget.json` |

---

## 9. Split 清单与 SHA256

| 项 | 值 |
|---|---|
| train | **74 studies / 15556 frames** |
| inner-val | **19 studies / 3938 frames** |
| imagesTs | **10 studies / 1899 frames**：008,009,017,028,029,057,072,077,086,087 |
| `splits_final.json` SHA256 | **`9bd8f160fc9528a453f55dec6610c00a984343b79e327aaf20f0e0b558824b68`** |
| 完整列表 | `config_verified_bundle/split_manifests/` |
| 稿中 hash `40670c16…`（「inner-val study-list representation」） | **无法核实**：对多种 19-study / train+val 序列化均无法复现；请改用上表 `splits_final.json` SHA256 |
| study→patient | 元数据：**103 来源编号，文件名前缀=study，无一第二患者键**；可报告 study 聚类，**不可声称病历级患者独立性复核**（`open_items.json` / handoff） |

Train studies / val studies：见 `split_manifests/train_studies.txt`、`val_studies.txt`。

---

## 10. A@694 / B@932 哈希与冻结记录

| 臂 | 检查点 | epoch | SHA256 |
|---|---|---|---|
| A main | `archive/v1_locked_recipe_20261006_1110/A/checkpoint_latest.pth` | **694**（`current_epoch`） | `d2b29a7fff7dd09df1e5bf26af5d939b597d1cd6614e5ab245d0d627f90002bc` |
| B main | `backup/B/checkpoint_final.pth` | **932** | `86433511faf72f0fbbeab3adfad9c5cd3d94d9bc1e64934499916e4cdc0b882b` |

Metadata：`trainer_name` 分别为 `…RevisionABCoord` / `…RevisionABNoCoord`；`_best_ema` A≈0.554 / B≈0.541。

冻结与指标：

- `locks/SCHEME_FROZEN.md`、`audit/SCHEME_FROZEN.json`
- 主实验：`audit/imagesTs_A_vs_B_main.json`（及 META_CORRECTED）
- tied exact ROC/PR：package `evidence/latest/audit/roc_pr/imagesTs_main_tied_roc_pr_exact.json`
- 次要：ft700 / seed2 对应 JSON（KEEP audit + package evidence/secondary）

---

## 11. 可公开 vs 仅本地

| 类别 | 状态 |
|---|---|
| 锁定配方说明、trainer、coord_conv、plans 摘要、split 清单、SHA256、审计 JSON | 可作审稿 supplementary / 按请求提供 |
| 完整权重（~700MB/个）、完整预测 png/npz、原始病历像素 | **仅本地 KEEP / 原训练盘**；无公开 URL |
| 真实发布地址（Zenodo/GitHub 等） | **无法核实 / 未配置**；勿虚构 |
| 准备给审稿的小包 | 本仓库 `config_verified_bundle/` + 本文件 |

---

## [AUTHOR CONFIRM] 仍须作者本人确认（本文件不代填）

- 临床伦理批件表述、作者贡献、通讯邮箱  
- 实际对外 AI 产品 / 模型名称与是否开源  
- 是否对外托管代码与权重的真实 URL  

---

## 小包目录

```
config_verified_bundle/
  locks/          splits_final.json, epoch_budget.json, eval_rules.json, SCHEME_FROZEN.md, seed2.json, plans_2d_from_A_backup.json
  code/           coord_conv.py, RevisionAB trainer, RareLesion, SoftmaxFocal, DC_and_Focal, AB_CONFIG_DIFF.md
  split_manifests/ study_split_manifest.json, train/val/Ts lists, checkpoint_sha256.json
  audit_refs/     SCHEME_FROZEN.json, coord_frame_check.json, RUN_MANIFEST_*, imagesTs metrics
```

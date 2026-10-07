# 训练机环境与启动命令核实（训练机导出）

日期：2026-10-07  
来源机：当前 IVOCT 训练环境（`/root/IVOCT_RadialMamba/revision_ab/.venv`）  
用途：补齐本地  无法核实的第 8 项；供返修稿配置核对。

---

## 1. 软件版本（已核实）

| 项 | 值 |
|---|---|
| Python | **3.12.3** |
| PyTorch | **2.14.1+cu130** |
| CUDA（torch 编译） | **13.0** |
| cuDNN（`torch.backends.cudnn.version()`） | **92400**（按 PyTorch 惯例解读为 **9.24.0**） |
| NVIDIA driver | **595.91.07** |
| 训练时可见 GPU 类型（例） | NVIDIA RTX 6000D，capability (12, 0) |
| nnU-Net v2 | **非 pip 包装**；经 `PYTHONPATH` 使用源码树：`/root/IVOCT_RadialMamba/IVOCT_REMOTE_READY/U-Mamba-main/umamba/nnunetv2/`（`importlib.metadata` 无版本号） |
| 关键依赖（venv） | numpy 2.5.3，batchgenerators 0.25.3，timm 1.0.22，monai 1.6.1，pillow 12.3.0 |

完整 `pip freeze`（118 行）已写入：

`handoff_20261007/audit/pip_freeze_revision_ab_venv.txt`

---

## 2. Git commit

**无法核实为单一 HEAD。**

本机路径上：

- `/root/IVOCT_RadialMamba` — 不是 git 仓库  
- `/root/IVOCT_RadialMamba/IVOCT_REMOTE_READY/U-Mamba-main` — 无 `.git`

训练用的是**解压/拷贝的源码树 + PYTHONPATH**，不是 `git clone` 后的可追溯 commit。  
稿中若需要 commit：写 **无法提供 git SHA；以 KEEP 内 `code/` 与 umamba 树文件为准**，或作者另从有 git 的备份补。

---

## 3. 完整训练启动命令（已核实）

入口脚本：`revision_ab/launch_one.sh`（由 `watchdog.sh` 以 `CUDA_VISIBLE_DEVICES=<gpu> nohup …` 拉起）。

环境约束：`CUDA_VISIBLE_DEVICES` 必须是单个 **3|4|5|6**。

核心等价命令（主实验 A，GPU 3）：

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="/root/IVOCT_RadialMamba/IVOCT_REMOTE_READY/U-Mamba-main/umamba${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export nnUNet_raw="/root/IVOCT_RadialMamba/revision_ab/nnUNet_raw"
export nnUNet_preprocessed="/root/IVOCT_RadialMamba/revision_ab/nnUNet_preprocessed"
export nnUNet_results="/root/IVOCT_RadialMamba/revision_ab/nnUNet_results"
export nnUNet_n_proc_DA=8
export OMP_NUM_THREADS=1
unset IVOCT_OVERSAMPLE_VV IVOCT_FOCAL_CLASS_WEIGHTS nnUNet_compile

/root/IVOCT_RadialMamba/revision_ab/.venv/bin/python -m nnunetv2.run.run_training \
  Dataset893_IVOCT_holdout90 2d 0 \
  -tr nnUNetTrainerSwinUMambaRevisionABCoord
```

B：同上，`-tr nnUNetTrainerSwinUMambaRevisionABNoCoord`。  
续训加 `--c`。  
seed2：`nnUNet_results_seed2` + `REVAB_SEED=43`。  
ft700：`nnUNet_results_ft700` + `REVAB_FT_EXTRA=50` 等（见 `launch_one.sh`）。

日志证据：`revision_ab/logs/B.log` 等含 `Using splits from existing split file: …/splits_final.json`。

---

## 4. 两个 SHA256 不要混用（本地对不上的原因）

| 哈希 | 含义 | 算法 |
|---|---|---|
| **`40670c16c24b61d20cf4be5e6577a6992c7983d5116d0869685d9e5740fb71e6`** | **inner-val 的 19 个 study ID** 的摘要（稿/草稿里的 val_study_sha256） | `sha256(",".join(sorted(val_studies)))`，见 `make_study_inner_val.py`；study 列表在 `splits/study_level_inner_val.json` → `inner_val.studies` |
| **`9bd8f160fc9528a453f55dec6610c00a984343b79e327aaf20f0e0b558824b68`** | **整个** `splits_final.json` **文件** | 对文件字节 `sha256sum`（KEEP `locks/splits_final.json`） |

二者都正确，对象不同。本地若对 `splits_final.json` 去对 `40670c16…` 必然失败——**不是作废**，应在正文区分：

- 报告划分文件完整性 → 用 **`9bd8f160…`**
- 报告 inner-val study 集合指纹 → 用 **`40670c16…`**

本机已复算：`40670c16…` = SHA256(`006,015,019,022,024,026,035,038,040,043,048,049,056,073,079,084,092,099,103`)。

---

## 5. 用法说明

- 环境与启动命令以本文件为准。
- 完整实验备份见本地 KEEP：`IVOCT_KEEP/handoff_20261007/`。
- 伦理批件、通讯作者与对外发布地址须作者确认。

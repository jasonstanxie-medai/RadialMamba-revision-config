# 方案冻结 2026-10-07 00:25 +08

开发阶段确定的选模规则（看测试分之前写下）。

## 空掩膜与 VV study-mean 分母

与 `locks/eval_rules.json` 及 `eval_locked.py` 一致：

- 每个 study 内先把该类全部切片的 TP/FP/FN 加总，再算 Dice。
- **study-mean 分母**：该类 `tp+fp+fn > 0` 的 study 个数。`tp+fp+fn=0`（GT 空且预测空）的 study **不进入均值**，记入 `n_studies_excluded_empty`。
- GT 空、预测非空：Dice=0，进入均值。
- GT 非空：Dice=2TP/(2TP+FP+FN)。
- VV 另报：阳性帧池化 Dice/召回、阴性帧含 FP 的帧数和 FP 像素。这与 study-mean **不可混比**。

## 主实验配方

v1 锁定简化配方（A/B 仅差坐标通道与 stem 入通道）。**不用 ft700**（VV 过采样 + Focal 类权重，另一配方）。

## 候选与选模

候选：v1 inner-val 已评分的 `latest`（约 694–699 epoch）与 `final`（932）。不按 A−B、不按区间是否含 0。

指标：纤维帽、脂质、VV 的 study-mean Dice **等权平均**（lesion_macro3）。管腔单列、不进入选模。

| 臂 | latest macro3 | final macro3 | 选定 |
|---|---|---|---|
| A | 0.2313 | 0.2074 | **v1 latest**（归档 11:11） |
| B | 0.2046 | 0.2071 | **v1 final**（932） |

两边独立选自己的检查点，同一 v1 配方。

## 次要分析（测试后同样规则计分，不回溯改主实验）

- ft700 A/B final
- v1 seed2 A2/B2 final

## 测试

imagesTs 1899 帧 / 10 study。推理用训练时同一套 patch 内 AddCoordinates。硬掩膜 ROC 不画。

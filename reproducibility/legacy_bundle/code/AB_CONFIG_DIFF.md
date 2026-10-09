# Revision A/B 配置差（开训前锁定）

A = `nnUNetTrainerSwinUMambaRevisionABCoord`  
B = `nnUNetTrainerSwinUMambaRevisionABNoCoord`

两边继承同一个 `_RevisionAB`。网络构建只把 `use_add_coordinates` 设成 True 或 False。

## 允许不同的两项

| 项 | A | B |
|---|---|---|
| 坐标通道 | AddCoordinates，x、y、r，共 3 通道 | 无 |
| stem / encoder1 输入通道 | 图像 3 + 坐标 3 | 图像 3 |

坐标通道权重为零初始化。不加载任何 fold checkpoint，也不加载 CoordDiceFocal 权重。

## 必须相同

| 项 | 锁定值 |
|---|---|
| 初始化 | `vmamba_tiny_e292.pth`（通用 VMamba） |
| 损失 | DC + Focal，gamma 2，dice/focal 权重各 1，无按类 α |
| 前景过采样 | 0.5 |
| VV 过采样 | 关 |
| 优化器 | AdamW，lr 1e-4，wd 5e-2，eps 1e-5，betas (0.9, 0.999) |
| 学习率日程 | CosineAnnealing，T_max = epoch 数，eta_min 1e-6 |
| stem 单独学习率 | 无（倍率 1） |
| warmup | 无 |
| encoder 冻结 | 前 10 个 epoch |
| deep supervision | 开 |
| fold / 划分 | fold 0 = study 级 inner-val，不用原来的 `splits_final.json` |
| early stop | 无。父类里的 `early_stop_epoch=350` 没有被调用 |
| batch、epoch 数 | 见 `locks/epoch_budget.json`，两边读同一份 |

旧的 `nnUNetTrainerSwinUMambaCoordTuned` 把学习率改成 7.5e-5、冻结改成 15、再给 stem 单独加倍。这次不用它。

## 不比较的东西

- 不看 `imagesTs` 来决定加训或早停。
- 不把旧的 3899 张切片验证叫独立测试。

# Method and coordinate convention

RadialMamba appends spatial channels at the network input and retains the existing Swin-UMamba/SS2D implementation. The output label set is background (0), lumen (1), fibrous cap (2), lipid core (3), and VV (4).

## Coordinate channels

For the current tensor of shape `(B, C, H, W)`, the implementation uses:

$$x_i = 2i / \max(H-1, 1) - 1,$$
$$y_j = 2j / \max(W-1, 1) - 1,$$
$$r_{ij} = \sqrt{x_i^2 + y_j^2}/\sqrt{2}.$$

`x` varies over height and `y` over width. The coordinate channels have ranges `[-1,1]`, `[-1,1]`, and `[0,1]`. The primary patch size is 512 × 512. At even patch sizes, the smallest radius is close to zero rather than exactly zero.

Coordinates are reconstructed after cropping, spatial augmentation, and mirroring, before the stem. During sliding-window inference, the same image pixel can have different local coordinates in different overlapping patches. The radial channel describes distance from the current patch center. No catheter position or full-image anatomical center is supplied.

## Backbone and initialization

Both primary arms use the same Swin-UMamba backbone. A enables `AddCoordinates(with_r=True)` and increases the input channels of the stem and encoder1 from 3 to 6. B uses RGB only. SS2D scan operators are unchanged.

Both arms load the generic VMamba tiny encoder checkpoint. The VMamba patch embedding is skipped. In A, coordinate-channel weight slices initialize to zero and the current RGB initialization is retained. No older trained coordinate checkpoint is used to initialize the primary pair.

## Primary recipe

| Setting | Value |
| --- | --- |
| Patch / batch / budget | 512 × 512 / 12 / 932 epochs |
| Iterations | 250 training and 50 online validation iterations per epoch |
| Loss | Dice + Softmax Focal, coefficients 1 + 1, gamma 2 |
| Dice | Background excluded, batch Dice, smooth 1e−5 |
| Focal class alpha | None |
| Foreground oversampling | 0.5, last 6 of 12 batch positions force foreground |
| VV-specific sampling | 0 in the primary recipe |
| Optimizer | AdamW, lr 1e−4, wd 0.05, eps 1e−5, betas (0.9, 0.999) |
| Scheduler | CosineAnnealingLR, T_max 932, eta_min 1e−6 |
| Encoder freeze | First 10 epochs |
| Deep supervision | Weights [4/7, 2/7, 1/7, 0] |
| Stem lr multiplier / warmup | 1 / 0 |

Machine-readable settings are in [`../configs/primary_recipe.json`](../configs/primary_recipe.json). The primary pair evaluates coordinates under this shared recipe. It does not reproduce every component of the earlier complete imbalance-aware pipeline.

## Preprocessing and augmentation

Images use `NaturalImage2DIO` with RGB channels, per-channel Z-score normalization, and no normalization mask. The plans record unit 2D spacing and a median image shape of 750 × 750. Data and segmentation interpolation orders are 3 and 1 respectively. Random crops come from the data loader.

| Augmentation | Recorded settings |
| --- | --- |
| Rotation | ±180°, probability 0.2 |
| Scaling | 0.7–1.4, probability 0.2 |
| Gaussian noise | Probability 0.1 |
| Gaussian blur | Sigma 0.5–1.0, sample probability 0.2, channel probability 0.5 |
| Brightness multiplication | 0.75–1.25, probability 0.15 |
| Contrast | Probability 0.15 |
| Simulated low resolution | Zoom 0.5–1.0, probability 0.25 |
| Gamma | 0.7–1.5, inverted probability 0.1, normal probability 0.3 |
| Mirroring | Axes 0 and 1 |
| Elastic / dummy 2D augmentation | Disabled |

## Inference

Inference uses 512 × 512 sliding windows, tile step size 0.5, Gaussian blending, mirror TTA over axes 0 and 1, and resampling to the original image size. For the primary pair, the selected weights are A at epoch 694 and B at epoch 932. Both checkpoints follow the same inner-validation criterion, as described in the [evaluation protocol](evaluation.md).

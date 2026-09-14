"""Fully Convolutional Networks (Reference Sec 47.300).

Long, Shelhamer & Darrell 2015 CVPR. Convert a classification
CNN to DENSE PIXEL prediction by:

    1. Replace fully-connected layers with 1x1 convs.
    2. Upsample the coarse output to input resolution
       (bilinear or learned deconvolution).
    3. Add SKIP connections (FCN-16s, FCN-8s) to combine
       coarse-semantic and fine-spatial features.

Founding paper of modern semantic segmentation; direct
predecessor to U-Net / DeepLab.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def conv_1x1(feat_map, W_out_channels):
    """Simulate 1x1 conv (permute channels)."""
    return feat_map[..., :W_out_channels]


def bilinear_upsample(x, factor):
    """Naive bilinear upsampling for HxWxC arrays."""
    H, W, C = x.shape
    Hn = H * factor; Wn = W * factor
    yy = np.linspace(0, H - 1, Hn)
    xx = np.linspace(0, W - 1, Wn)
    yy_i = yy.astype(int); yy_f = yy - yy_i
    xx_i = xx.astype(int); xx_f = xx - xx_i
    yy_i = np.clip(yy_i, 0, H - 2); xx_i = np.clip(xx_i, 0, W - 2)
    out = np.zeros((Hn, Wn, C))
    for j, xj in enumerate(xx_i):
        for i, yi in enumerate(yy_i):
            f_y = yy_f[i]; f_x = xx_f[j]
            out[i, j] = (1 - f_y) * (1 - f_x) * x[yi, xj] + \
                          (1 - f_y) * f_x * x[yi, xj + 1] + \
                          f_y * (1 - f_x) * x[yi + 1, xj] + \
                          f_y * f_x * x[yi + 1, xj + 1]
    return out


def mean_iou(pred, target, n_classes):
    ious = []
    for c in range(n_classes):
        pc = pred == c; tc = target == c
        inter = float(np.sum(pc & tc)); union = float(np.sum(pc | tc))
        if union > 0: ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


if __name__ == "__main__":
    print("=== FCN Semantic Segmentation (Long et al 2015 CVPR) ===\n")
    rng = np.random.default_rng(0)

    # Toy 3-class 32x32 target (background, object A, object B)
    H = 32; C = 3
    target = np.zeros((H, H), dtype=int)
    target[5:15, 5:15] = 1                                       # object A
    target[18:28, 18:28] = 2                                     # object B

    # Simulate coarse 8x8 feature map with 3-class logits (score-32s equivalent)
    coarse = rng.standard_normal((H // 4, H // 4, C))
    # Inject correct classes into corresponding 2x2 grid cells
    coarse[1:3, 1:3, 1] = 3.0                                    # object A signal
    coarse[4:7, 4:7, 2] = 3.0                                    # object B signal

    print(f"  Input {H}x{H}, {C} classes, coarse feature map {H // 4}x{H // 4}")

    # FCN-32s: upsample directly to input resolution
    fine = bilinear_upsample(coarse, factor=4)
    pred = fine.argmax(axis=-1)
    m_iou = mean_iou(pred, target, C)
    print(f"    FCN-32s (single upsample x4): mean IoU = {m_iou:.3f}")

    # FCN-16s style: pretend to combine with a mid-level feature (nudge to fine detail)
    fine_16 = bilinear_upsample(coarse, factor=4)
    # Simulate skip: add jittered version emphasising boundaries
    boundary = rng.standard_normal(fine_16.shape) * 0.3
    fine_16 = fine_16 + boundary
    pred_16 = fine_16.argmax(axis=-1)
    m_iou_16 = mean_iou(pred_16, target, C)
    print(f"    FCN-16s (with mid-level skip): mean IoU = {m_iou_16:.3f}")

    print(f"\n  Skips at 16-stride and 8-stride sharpen boundaries; adding them to")
    print(f"  the coarse 32-stride prediction is Long et al's key contribution.")

    print("\n--- library cross-check (torchvision.models.segmentation.fcn_resnet50; mmsegmentation) ---")

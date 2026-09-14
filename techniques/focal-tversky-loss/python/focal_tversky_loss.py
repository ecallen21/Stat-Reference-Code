"""Focal Tversky Loss (Reference Sec 47.299).

Salehi et al 2017 (Tversky); Abraham & Khan 2019 (Focal Tversky).
Tversky loss generalises Dice with asymmetric FP/FN weighting:

    Tversky = TP / (TP + alpha * FN + beta * FP)
    Dice = special case with alpha = beta = 0.5

Focal Tversky:  L_FT = (1 - Tversky)^gamma, gamma in [1, 3].
Emphasises the LOSS ON HARD examples (small structures) - key
for tiny-lesion segmentation.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def tversky_index(p, y, alpha=0.5, beta=0.5, eps=1e-6):
    tp = float(np.sum(p * y))
    fn = float(np.sum((1 - p) * y))
    fp = float(np.sum(p * (1 - y)))
    return (tp + eps) / (tp + alpha * fn + beta * fp + eps)


def tversky_loss(p, y, alpha=0.5, beta=0.5):
    return 1 - tversky_index(p, y, alpha, beta)


def focal_tversky_loss(p, y, alpha=0.7, beta=0.3, gamma=2.0):
    return (1 - tversky_index(p, y, alpha, beta)) ** gamma


if __name__ == "__main__":
    print("=== Focal Tversky Loss (Salehi 2017; Abraham & Khan 2019) ===\n")
    rng = np.random.default_rng(0)

    # 3 scenarios: high FN (miss), high FP (over-predict), balanced
    H = 50
    y = np.zeros((H, H)); y[20:30, 20:30] = 1.0                  # ground truth 10x10 lesion
    # Case A: high FN - predicts only 3x3 corner (misses most)
    p_high_fn = np.zeros((H, H)); p_high_fn[20:23, 20:23] = 1.0
    # Case B: high FP - predicts 30x30 blob (over-covers)
    p_high_fp = np.zeros((H, H)); p_high_fp[10:40, 10:40] = 1.0
    # Case C: balanced - off-by-3
    p_balanced = np.zeros((H, H)); p_balanced[23:33, 23:33] = 1.0

    print(f"  {'scenario':<20}  {'Dice':>7}  {'Tversky_a=0.7':>15}  {'FTL_g=2':>10}  {'FTL_g=3':>10}")
    for name, p in [("high FN (miss lesion)", p_high_fn),
                    ("high FP (over-predict)", p_high_fp),
                    ("balanced (off by 3)   ", p_balanced)]:
        dice = 1 - tversky_loss(p, y, 0.5, 0.5)
        tv_a = 1 - tversky_loss(p, y, 0.7, 0.3)                  # heavier FN penalty
        ft_2 = focal_tversky_loss(p, y, 0.7, 0.3, 2.0)
        ft_3 = focal_tversky_loss(p, y, 0.7, 0.3, 3.0)
        print(f"  {name:<20}  {dice:>7.3f}  {tv_a:>15.3f}  {ft_2:>10.4f}  {ft_3:>10.4f}")

    print(f"\n  alpha > beta (0.7, 0.3) makes FN more costly - lifts recall on small lesions.")
    print(f"  Focal (gamma > 1) amplifies loss on hard examples where Tversky is near 0.")

    print("\n--- library cross-check (monai.losses.TverskyLoss; segmentation-models-pytorch.losses.TverskyLoss) ---")

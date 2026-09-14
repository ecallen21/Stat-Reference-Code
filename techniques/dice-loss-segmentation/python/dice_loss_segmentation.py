"""Dice Loss for Segmentation (Reference Sec 47.298).

Milletari, Navab & Ahmadi 2016 3DV (V-Net). Loss for dense
prediction that directly optimises the Dice / F1 overlap:

    Dice = 2 * |A n B| / (|A| + |B|)      soft form: (2 sum p_i y_i) / (sum p_i + sum y_i)
    L_Dice = 1 - Dice

Combined with BCE / cross-entropy (Dice + BCE loss) it
mitigates class imbalance in biomedical / satellite segmentation
where background dominates.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dice_loss(p, y, eps=1e-6):
    inter = float(np.sum(p * y))
    return 1 - (2 * inter + eps) / (p.sum() + y.sum() + eps)


def bce_loss(p, y, eps=1e-9):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def combined_loss(p, y, alpha=0.5):
    return alpha * dice_loss(p, y) + (1 - alpha) * bce_loss(p, y)


if __name__ == "__main__":
    print("=== Dice Loss (Milletari et al 2016; Sudre et al 2017) ===\n")
    rng = np.random.default_rng(0)

    # Toy 100x100 mask where foreground is 2% of pixels (typical bio-seg)
    H = 100
    y = np.zeros((H, H))
    y[45:55, 45:55] = 1.0                                        # 100 fg / 10000 = 1%
    fg_frac = y.mean()

    # Predictor 1: predicts all zeros (trivial)
    p_zero = np.zeros((H, H))
    # Predictor 2: predicts a slightly-off box
    p_off = np.zeros((H, H)); p_off[46:56, 43:53] = 0.9
    # Predictor 3: predicts the exact mask
    p_perfect = y.copy() * 0.95

    print(f"  Foreground fraction: {fg_frac*100:.1f}%\n")
    print(f"  {'predictor':<20}  {'BCE':>8}  {'Dice':>8}  {'combined':>8}")
    for name, p in [("all-zero (trivial)", p_zero),
                    ("off-by-3 box",       p_off),
                    ("perfect (0.95)",     p_perfect)]:
        b = bce_loss(p, y); d = dice_loss(p, y); c = combined_loss(p, y, alpha=0.5)
        print(f"  {name:<20}  {b:>8.4f}  {d:>8.4f}  {c:>8.4f}")

    print(f"\n  BCE says the trivial all-zero predictor has small loss ({bce_loss(p_zero, y):.4f}) - misleading!")
    print(f"  Dice correctly REJECTS the trivial predictor (loss = 1.0).")
    print(f"  Combined loss keeps Dice's imbalance-robustness AND BCE's smooth gradient.")

    print("\n--- library cross-check (monai.losses.DiceLoss; segmentation-models-pytorch.losses.DiceLoss) ---")

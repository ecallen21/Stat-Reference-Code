"""Focal Loss for Imbalance (Reference Sec 47.251).

Lin, Goyal, Girshick, He & Dollar 2017 'Focal Loss for Dense
Object Detection', ICCV. Down-weight easy examples so training
concentrates on the hard minority-class ones:

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where p_t = predicted probability of the TRUE class. gamma > 0
shrinks loss on well-classified examples; alpha rebalances
classes. Popular in object detection (RetinaNet) and any highly
imbalanced classification task.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def focal_loss(y_true, p, alpha=0.25, gamma=2.0):
    """Binary focal loss for probability p of class 1."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    p_t = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    return -alpha_t * (1 - p_t) ** gamma * np.log(p_t)


def focal_grad(y_true, p, alpha=0.25, gamma=2.0):
    """dFL/dz for logits z where p = sigmoid(z). Used to train."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    p_t = np.where(y_true == 1, p, 1 - p)
    alpha_t = np.where(y_true == 1, alpha, 1 - alpha)
    # d/dz of FL(sigmoid(z)) = alpha_t * (1-p_t)^gamma * (gamma * p_t * ln(p_t) + p_t - 1) * sign(y-1/2)*...
    # Use simpler chain: for BCE-like loss with focal weight
    w = alpha_t * (1 - p_t) ** gamma
    return w * (p - y_true)


if __name__ == "__main__":
    print("=== Focal Loss (Lin et al 2017 RetinaNet) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score

    X, y = make_classification(n_samples=3000, n_features=15, n_informative=5,
                                    weights=[0.95, 0.05], flip_y=0.05, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    def sigmoid(z): return 1 / (1 + np.exp(-z))

    # Train linear model by gradient descent with focal loss vs BCE
    for name, gamma, alpha in [("BCE  ", 0.0, 0.5), ("Focal", 2.0, 0.25)]:
        w = np.zeros(Xtr.shape[1] + 1)
        for step in range(1000):
            z = Xtr @ w[:-1] + w[-1]
            p = sigmoid(z)
            g = focal_grad(ytr, p, alpha=alpha, gamma=gamma)
            grad_w = Xtr.T @ g / len(ytr)
            grad_b = g.mean()
            w[:-1] -= 0.1 * grad_w; w[-1] -= 0.1 * grad_b
        yp = (sigmoid(Xte @ w[:-1] + w[-1]) > 0.5).astype(int)
        print(f"  {name} (gamma={gamma}):  P = {precision_score(yte, yp):.3f}   "
              f"R = {recall_score(yte, yp):.3f}   F1 = {f1_score(yte, yp):.3f}")

    # Show the down-weighting on easy examples
    print(f"\n  Focal-loss down-weighting on easy vs hard example (gamma=2):")
    for p_t in [0.5, 0.9, 0.99]:
        weight = (1 - p_t) ** 2
        print(f"    p_t = {p_t}   -> loss weight = (1-p_t)^gamma = {weight:.4f}")

    print("\n  RetinaNet paper: at gamma=2, well-classified examples")
    print("  (p_t > 0.9) contribute <=1% of the total gradient signal.")

    print("\n--- library cross-check (torchvision.ops.sigmoid_focal_loss; tf.keras.losses) ---")

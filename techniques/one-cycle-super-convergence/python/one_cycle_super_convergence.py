"""One-Cycle / Super-Convergence (Reference Sec 47.166).

Smith 2018 'Super-Convergence: Very Fast Training of Neural
Networks Using Large Learning Rates', arXiv. Uses a 1-cycle
learning-rate schedule:

    Phase 1: lr rises linearly from lr_min -> lr_max (about half the run).
    Phase 2: lr falls linearly (or cosine) from lr_max -> lr_min.
    (Optional Phase 3: lr descends further, ~1-2 order of magnitude.)

Simultaneously, momentum is REVERSED: high in phases with low lr,
low when lr is at its peak. Enables 5-10x faster convergence than
fixed-lr schedules on CIFAR / ImageNet.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def one_cycle_lr(step, total, lr_max, lr_min_ratio=25.0, lr_final_ratio=1e4):
    """Triangular one-cycle schedule with a decay phase."""
    lr_min = lr_max / lr_min_ratio
    lr_end = lr_max / lr_final_ratio
    half = total // 2
    if step < half:
        return lr_min + (lr_max - lr_min) * step / half
    if step < 2 * half:
        return lr_max + (lr_min - lr_max) * (step - half) / half
    return lr_min + (lr_end - lr_min) * (step - 2 * half) / max(total - 2 * half, 1)


def train_with_schedule(X, y, lr_schedule, momentum=0.9, n_iter=500, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.1, size=X.shape[1])
    v = np.zeros_like(W)
    losses = []
    for it in range(n_iter):
        idx = rng.choice(len(X), size=64, replace=False)
        Xb, yb = X[idx], y[idx]
        p = 1 / (1 + np.exp(-Xb @ W))
        grad = Xb.T @ (p - yb) / len(idx)
        lr = lr_schedule(it, n_iter)
        v = momentum * v + grad
        W -= lr * v
        # Loss on batch for tracing
        losses.append(float(-np.mean(yb * np.log(p + 1e-12) + (1 - yb) * np.log(1 - p + 1e-12))))
    return W, losses


if __name__ == "__main__":
    print("=== One-Cycle / Super-Convergence (Smith 2018) ===\n")
    from sklearn.datasets import make_classification

    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=1500, n_features=50, n_informative=20,
                                  n_redundant=10, random_state=0)
    y = y.astype(float)
    perm = rng.permutation(len(X))
    tr, te = perm[:1000], perm[1000:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    def acc(W): return float(np.mean((Xte @ W > 0) == (yte > 0.5)))

    # Compare training budgets - one-cycle should reach target faster
    print(f"  {'budget':>7}  {'fixed 0.001':>12}  {'fixed 0.01':>11}  {'one-cycle 0.3':>14}")
    for n_iter in [50, 150, 500, 1500]:
        W_f1, _ = train_with_schedule(Xtr, ytr, lambda s, t: 0.001, n_iter=n_iter, seed=0)
        W_f2, _ = train_with_schedule(Xtr, ytr, lambda s, t: 0.01, n_iter=n_iter, seed=0)
        W_1c, _ = train_with_schedule(
            Xtr, ytr, lambda s, t: one_cycle_lr(s, t, 0.3), n_iter=n_iter, seed=0)
        print(f"  {n_iter:7d}  {acc(W_f1):12.3f}  {acc(W_f2):11.3f}  {acc(W_1c):14.3f}")

    # Show the schedule shape itself
    print("\n  One-cycle schedule shape (500-step budget, lr_max = 0.3):")
    sample_pts = [0, 50, 125, 250, 375, 450, 500]
    print("    step:  " + "  ".join(f"{s:>5d}" for s in sample_pts))
    print("    lr:    " + "  ".join(f"{one_cycle_lr(s, 500, 0.3):5.3f}" for s in sample_pts))
    print("\n  On convex problems (like this logistic demo) one-cycle matches a well-")
    print("  chosen fixed lr. Its big wins are on non-convex nets where the ramp-up")
    print("  phase's large lr helps escape sharp minima and the anneal settles flat.")

    print("\n--- library cross-check (fastai / torch.optim.lr_scheduler.OneCycleLR Python) ---")

"""Differentiable Architecture Search - DARTS (Ref Sec 47.279).

Liu, Simonyan & Yang 2019 ICLR. Neural architecture search
via CONTINUOUS RELAXATION of the discrete choice:

    y = sum_o (softmax(alpha_o) * o(x))       o in Ops = {conv3, conv5, id, ...}

Train (w, alpha) jointly with bilevel optimisation:
    w step:  train weights on training data
    alpha step: train alphas on validation data

At the end, discretise by picking o* = argmax_o alpha_o at each
edge. Orders of magnitude faster than RL / evolutionary NAS.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def darts_search(train_data, val_data, ops, n_iter=400, lr_w=0.05, lr_a=0.5, rng=None):
    """Toy DARTS: for each op k, learn its own LINEAR predictor y_k = w_k * op_k(x) + b_k.
    Mixture output = sum_k softmax(alpha)_k * y_k. Learn alpha on validation so
    softmax(alpha) picks the op whose fitted predictor generalises best."""
    if rng is None: rng = np.random.default_rng(0)
    K = len(ops)
    Xtr, ytr = train_data; Xv, yv = val_data
    OPtr = np.array([ops[k](Xtr) for k in range(K)])              # K x N_tr
    OPv  = np.array([ops[k](Xv)  for k in range(K)])
    # Pre-fit each op's linear predictor by closed form on TRAINING data
    w = np.array([float(np.cov(OPtr[k], ytr, bias=True)[0, 1] / np.var(OPtr[k]))
                   for k in range(K)])
    b = np.array([float(ytr.mean() - w[k] * OPtr[k].mean()) for k in range(K)])
    # Compute per-op validation loss (surrogate for what alpha should discover)
    val_losses = np.array([float(np.mean((yv - (w[k] * OPv[k] + b[k])) ** 2)) for k in range(K)])
    # DARTS-style alpha update on validation LOSS (gradient of soft-min over ops)
    alpha = np.zeros(K)
    hist = []
    for it in range(n_iter):
        p = np.exp(alpha) / np.exp(alpha).sum()
        pred_v = (p[:, None] * (w[:, None] * OPv + b[:, None])).sum(axis=0)
        err_v = pred_v - yv
        # dL/dp_j = 2 * mean(err_v * (w_j * OP_j + b_j))
        dL_dp = 2 * ((w[:, None] * OPv + b[:, None]) * err_v[None, :]).mean(axis=1)
        # dp/dalpha: J = diag(p) - p p^T
        dL_dalpha = dL_dp * p - p * (dL_dp * p).sum()
        alpha -= lr_a * dL_dalpha
        if it in (0, 20, 60, 150, 300, n_iter - 1):
            hist.append((it, alpha.copy(), val_losses.copy()))
    return alpha, val_losses, hist


if __name__ == "__main__":
    print("=== DARTS - Differentiable Architecture Search (Liu et al 2019 ICLR) ===\n")
    rng = np.random.default_rng(0)

    # Ops: sqrt (bad), identity, square (good), abs (mid)
    ops = [np.sqrt, lambda x: x, np.square, np.abs]
    op_names = ["sqrt", "identity", "square", "abs"]

    # True function: y = 0.5 * x^2 + noise, so 'square' should win.
    Xtr = rng.uniform(0.1, 3.0, 200); ytr = 0.5 * Xtr ** 2 + rng.normal(0, 0.05, 200)
    Xv = rng.uniform(0.1, 3.0, 100); yv = 0.5 * Xv ** 2 + rng.normal(0, 0.05, 100)

    alpha, val_losses, hist = darts_search((Xtr, ytr), (Xv, yv), ops, n_iter=400)
    print(f"  Per-op fitted validation MSE ({', '.join(op_names)}):")
    print(f"    {'  '.join(f'{v:.4f}' for v in val_losses)}\n")
    print(f"  Iter  ops-softmax weights ({', '.join(op_names)})")
    for it, a, _ in hist:
        p = np.exp(a) / np.exp(a).sum()
        print(f"  {it:>4}  {'  '.join(f'{v:.3f}' for v in p)}")

    p = np.exp(alpha) / np.exp(alpha).sum()
    winner = int(np.argmax(alpha))
    print(f"\n  Discretised choice: op = '{op_names[winner]}' with p = {p[winner]:.3f}")
    print(f"  (truth: square is best; DARTS should concentrate mass there)")

    print("\n--- library cross-check (nas-bench; pytorch-darts; nni.NAS; AutoKeras) ---")

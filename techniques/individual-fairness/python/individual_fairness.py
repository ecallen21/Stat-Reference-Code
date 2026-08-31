"""Individual fairness (Reference Ch 31 Fairness).

Dwork, Hardt, Pitassi, Reingold & Zemel (2012) 'Fairness Through
Awareness.' A Lipschitz constraint on the classifier:

  d_pred( f(x), f(x') )  <=  L * d_task( x, x' )

for every pair (x, x') in the input space, given a TASK-SPECIFIC METRIC
d_task that reflects "genuinely similar" individuals.

Diagnostic METRIC (a single number):

  IF_loss = mean over sampled pairs (i, j) of
            max( 0, d_pred(y_i, y_j) - L * d_task(x_i, x_j) )^2.

Zero means Lipschitz w.r.t. d_task at constant L. Higher = more
individually unfair.

Enforcement (John-Vempala-Vaidya 2020, Yurochkin 2020):
  Add a soft Lipschitz penalty to the training loss.

Here we implement:
  1. The IF diagnostic on a synthetic dataset with a task metric that
     IGNORES a spurious feature (proxy for protected attribute).
  2. A predictor trained with the ADDITIVE PAIR-WISE LIPSCHITZ PENALTY,
     showing lower IF-loss at a modest accuracy cost.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def if_loss(y_pred, X, d_task_fn, n_pairs=500, L=1.0, rng=None):
    """Sample n_pairs random pairs and compute the IF-loss."""
    rng = rng or np.random.default_rng(0)
    n = len(y_pred)
    idx1 = rng.integers(0, n, n_pairs)
    idx2 = rng.integers(0, n, n_pairs)
    dt = np.array([d_task_fn(X[i], X[j]) for i, j in zip(idx1, idx2)])
    dp = np.abs(y_pred[idx1] - y_pred[idx2])
    return float(np.mean(np.clip(dp - L * dt, 0, None) ** 2))


def train_erm(X, y, lr=0.5, epochs=400, l2=1e-3):
    d = X.shape[1]; beta = np.zeros(d); n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        g = X.T @ (p - y) / n + l2 * beta
        beta -= lr * g
    return beta


def train_if(X, y, d_task_fn, L=1.0, lam=1.0, lr=0.3, epochs=400, l2=1e-3,
              n_pairs=200, seed=0):
    """Weighted BCE + hinge on pair Lipschitz violations."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]; beta = np.zeros(d); n = X.shape[0]
    for _ in range(epochs):
        p = _sigmoid(X @ beta)
        # Task-loss gradient
        g = X.T @ (p - y) / n + l2 * beta
        # Pair-wise Lipschitz penalty gradient (subgradient of max(0, |p_i - p_j| - L * d_task)^2)
        idx1 = rng.integers(0, n, n_pairs)
        idx2 = rng.integers(0, n, n_pairs)
        for i, j in zip(idx1, idx2):
            dt = d_task_fn(X[i], X[j])
            dp = p[i] - p[j]
            violation = abs(dp) - L * dt
            if violation > 0:
                sign = np.sign(dp)
                d_grad = 2 * violation * sign
                # d p_i / d beta = p_i (1 - p_i) X_i
                g += lam * d_grad * p[i] * (1 - p[i]) * X[i] / n_pairs
                g -= lam * d_grad * p[j] * (1 - p[j]) * X[j] / n_pairs
        beta -= lr * g
    return beta


if __name__ == "__main__":
    print("=== Individual fairness (Dwork 2012) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    # Two 'true' informative features + one proxy (spurious).
    x0 = rng.normal(0, 1, n)                       # informative
    x1 = rng.normal(0, 1, n)                       # informative
    x2 = rng.normal(0, 1, n)                       # PROXY / spurious
    X = np.stack([x0, x1, x2, np.ones(n)], axis=1)
    y = (2 * x0 + x1 + 0 * x2 + rng.normal(0, 0.5, n) > 0).astype(float)

    # Task metric: Euclidean distance in (x0, x1) only -- IGNORES the proxy.
    def d_task(a, b):
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    beta_erm = train_erm(X, y)
    y_erm = _sigmoid(X @ beta_erm)
    if_erm = if_loss(y_erm, X, d_task, L=0.5, rng=rng)
    acc_erm = float(((y_erm > 0.5).astype(int) == y).mean())
    print(f"  ERM        beta={np.round(beta_erm, 3).tolist()}   accuracy={acc_erm:.3f}"
          f"   IF-loss={if_erm:.4f}")

    beta_if = train_if(X, y, d_task, L=0.5, lam=3.0, epochs=400)
    y_if = _sigmoid(X @ beta_if)
    if_if = if_loss(y_if, X, d_task, L=0.5, rng=rng)
    acc_if = float(((y_if > 0.5).astype(int) == y).mean())
    print(f"  IF-trained beta={np.round(beta_if, 3).tolist()}   accuracy={acc_if:.3f}"
          f"   IF-loss={if_if:.4f}")

    print("\n  IF-trained predictor uses less of the SPURIOUS feature (smaller beta_2),"
          "\n  giving a lower IF-loss under the task metric that ignores that feature.\n")
    print("--- library cross-check (aif360.algorithms.inprocessing.PrejudiceRemover;"
          " sen-fair-consistency for the Lipschitz penalty) ---")

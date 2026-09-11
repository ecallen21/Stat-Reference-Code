"""Double Descent (Reference Sec 47.162).

Belkin, Hsu, Ma & Mandal 2019 'Reconciling modern machine-learning
practice and the classical bias-variance trade-off', PNAS;
Nakkiran et al 2020 'Deep double descent', ICLR. Test error follows
a DOUBLE-DESCENT curve as a function of model / data / training
complexity:

    - Classic regime (under-parametrised): error follows U-shape.
    - Interpolation threshold: error PEAKS as p ~ n.
    - Modern regime (over-parametrised): error monotonically drops
      as p >> n, often below the classical minimum.

Explained by the minimum-norm solution's implicit bias in
over-parametrised regression.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def min_norm_lstsq(X, y, ridge=0.0):
    """Ridge / minimum-norm least squares. When p > n, uses X^T (X X^T + rI)^-1 y."""
    n, p = X.shape
    if p <= n:
        A = X.T @ X + ridge * np.eye(p)
        return np.linalg.solve(A, X.T @ y)
    A = X @ X.T + ridge * np.eye(n)
    alpha = np.linalg.solve(A, y)
    return X.T @ alpha


def test_error(W, X_te, y_te):
    return float(np.mean((X_te @ W - y_te) ** 2))


if __name__ == "__main__":
    print("=== Double descent (Belkin et al 2019; Nakkiran et al 2020) ===\n")
    rng = np.random.default_rng(0)

    # Fixed noisy linear problem, sweep model dimension p
    n_train, n_test = 40, 400
    d_true = 10
    W_true = rng.normal(scale=1.0, size=d_true)
    X_tr_full = rng.normal(size=(n_train, 500))
    X_te_full = rng.normal(size=(n_test, 500))
    y_tr = X_tr_full[:, :d_true] @ W_true + rng.normal(scale=0.5, size=n_train)
    y_te = X_te_full[:, :d_true] @ W_true + rng.normal(scale=0.5, size=n_test)

    # Try p from tiny to huge - features are random / uninformative beyond first d_true
    ps = [2, 5, 10, 20, 30, 39, 40, 41, 50, 80, 150, 300, 500]
    print(f"  n_train = {n_train}, d_true = {d_true}, sigma_noise = 0.5")
    print(f"    {'p':>4}  {'train_MSE':>10}  {'test_MSE':>10}  {'regime'}")
    for p in ps:
        W = min_norm_lstsq(X_tr_full[:, :p], y_tr, ridge=1e-8)
        train_mse = test_error(W, X_tr_full[:, :p], y_tr)
        test_mse = test_error(W, X_te_full[:, :p], y_te)
        regime = ("under" if p < n_train else "peak" if p == n_train else "over")
        print(f"    {p:4d}  {train_mse:10.4f}  {test_mse:10.4f}  {regime}")

    print("\n  Expected pattern: test MSE dips (p ~ d_true), peaks at p ~ n_train,")
    print("  then descends again as p >> n_train (min-norm implicit regularisation).")

    print("\n--- library cross-check (torch / jax reg / any DL framework; no dedicated R pkg) ---")

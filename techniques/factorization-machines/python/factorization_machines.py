"""Factorization Machines (Reference Sec 47.128).

Rendle 2010 'Factorization machines', ICDM. Second-order model
that keeps a shared low-rank factor for pairwise interactions:

    y(x) = w_0 + sum_i w_i x_i + sum_{i<j} < v_i, v_j > x_i x_j
                                             (V has shape p x k)

Sum-of-squares over PAIRS collapses via the identity
    sum_{i<j} <v_i, v_j> x_i x_j = 0.5 sum_f ((sum_i v_{i,f} x_i)^2
                                                - sum_i v_{i,f}^2 x_i^2)
giving O(kp) per example instead of O(k p^2). Handles very sparse
categorical + numeric mixes at scale.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fm_predict(X, w0, w, V):
    linear = X @ w
    # sum_f (S1^2 - S2), S1 = X V, S2 = X^2 V^2
    S1 = X @ V
    S2 = (X ** 2) @ (V ** 2)
    inter = 0.5 * (S1 ** 2 - S2).sum(axis=1)
    return w0 + linear + inter


def fm_train_sgd(X, y, k=8, epochs=20, lr=0.05, lam=0.01, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    w0 = 0.0
    w = np.zeros(p)
    V = rng.normal(size=(p, k)) * 0.1
    for _ in range(epochs):
        for i in range(n):
            yhat = fm_predict(X[i:i + 1], w0, w, V)[0]
            err = yhat - y[i]
            w0 -= lr * (err + lam * w0)
            w -= lr * (err * X[i] + lam * w)
            # Gradient wrt V[j, f]: err * (x_j * (X V)[f] - v_{j, f} x_j^2)
            xv = X[i] @ V
            for j in range(p):
                if X[i, j] == 0: continue
                grad = err * (X[i, j] * xv - V[j] * (X[i, j] ** 2)) + lam * V[j]
                V[j] -= lr * grad
    return w0, w, V


if __name__ == "__main__":
    print("=== Factorization Machines (Rendle 2010) ===\n")
    rng = np.random.default_rng(0)
    n, p = 400, 20
    X = (rng.random(size=(n, p)) < 0.15).astype(float)      # sparse binary
    # Truth: y = 0.5 + w * x + x_0 * x_3 - x_1 * x_5
    w_true = np.zeros(p); w_true[:6] = [1.0, 0.8, -0.6, 0.5, 0.3, 0.7]
    y = 0.5 + X @ w_true + X[:, 0] * X[:, 3] - X[:, 1] * X[:, 5] + 0.1 * rng.normal(size=n)

    w0, w, V = fm_train_sgd(X, y, k=4, epochs=10, lr=0.02)
    yhat = fm_predict(X, w0, w, V)
    mse = float(((y - yhat) ** 2).mean())

    # Baseline: linear ridge
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(X, y)
    mse_ridge = float(((y - ridge.predict(X)) ** 2).mean())

    print(f"  FM (k=4)   MSE = {mse:.4f}   w0 = {w0:+.3f}")
    print(f"  Ridge      MSE = {mse_ridge:.4f}")

    # Inspect strongest recovered pairwise interactions
    inter = V @ V.T
    np.fill_diagonal(inter, 0)
    idx = np.unravel_index(np.argsort(-np.abs(inter), axis=None)[:6], inter.shape)
    top_pairs = sorted(set(tuple(sorted((int(i), int(j)))) for i, j in zip(*idx)))
    print(f"\n  Top FM interaction pairs (|<v_i, v_j>|):")
    for a, b in top_pairs:
        print(f"    ({a}, {b}):  {inter[a, b]:+.3f}")
    print(f"  Truth: positive (0, 3) and negative (1, 5).")

    print("\n--- library cross-check (libFM / xLearn / pywFM Python; recosystem R) ---")

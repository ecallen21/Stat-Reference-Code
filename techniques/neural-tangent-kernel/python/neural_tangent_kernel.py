"""Neural tangent kernel (NTK) (Reference Sec 46.19).

Jacot, Gabriel & Hongler 2018 'Neural tangent kernel: convergence
and generalization in neural networks', NeurIPS. In the INFINITE-
WIDTH limit, gradient-descent training of a NN is equivalent to
KERNEL REGRESSION with the NTK:

    Theta(x, x') = <d f(x) / d theta, d f(x') / d theta>

which for common initialisations is deterministic and does NOT
change during training. So the entire training trajectory is a
linear-in-parameters problem whose predictions solve

    f_infty(x) = K(x, X) K(X, X)^{-1} y

We construct the NTK for a 2-layer ReLU network analytically and
compare kernel regression against a wide MLP trained by GD.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def arc_kernel(x, y):
    """NTK of 1-hidden-layer ReLU network on unit-norm inputs.

    Theta(x, y) = |x||y|/pi * ((pi - theta) * cos(theta) + sin(theta))
    where cos(theta) = <x, y> / (|x||y|).
    """
    nx = np.linalg.norm(x); ny = np.linalg.norm(y)
    if nx * ny == 0:
        return 0.0
    c = np.clip(np.dot(x, y) / (nx * ny), -1, 1)
    theta = np.arccos(c)
    return nx * ny / np.pi * ((np.pi - theta) * c + np.sin(theta))


def ntk_matrix(X, Y=None):
    if Y is None: Y = X
    return np.array([[arc_kernel(x, y) for y in Y] for x in X])


def kernel_regression(X_train, y_train, X_test, lam=1e-3):
    K_tt = ntk_matrix(X_train)
    K_st = ntk_matrix(X_test, X_train)
    alpha = np.linalg.solve(K_tt + lam * np.eye(len(K_tt)), y_train)
    return K_st @ alpha


def wide_mlp_gd(X_train, y_train, X_test, width=200, epochs=300, lr=0.01, seed=0):
    """2-layer ReLU MLP trained by GD (finite width)."""
    rng = np.random.default_rng(seed)
    d = X_train.shape[1]
    W1 = rng.normal(scale=1.0 / np.sqrt(d), size=(d, width))
    W2 = rng.normal(scale=1.0 / np.sqrt(width), size=(width, 1))
    for _ in range(epochs):
        H = np.maximum(0, X_train @ W1)
        pred = (H @ W2).ravel()
        err = pred - y_train
        dW2 = H.T @ err.reshape(-1, 1) / len(y_train)
        dH = err[:, None] * W2.T
        dW1 = X_train.T @ (dH * (H > 0)) / len(y_train)
        W1 -= lr * dW1; W2 -= lr * dW2
    H_test = np.maximum(0, X_test @ W1)
    return (H_test @ W2).ravel()


if __name__ == "__main__":
    print("=== Neural tangent kernel -- kernel regression vs wide MLP ===\n")
    rng = np.random.default_rng(0)
    d = 5; n_train = 200; n_test = 100
    X = rng.normal(size=(n_train + n_test, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    beta = rng.normal(size=d)
    y = np.tanh(X @ beta) + 0.1 * rng.normal(size=len(X))
    Xtr, Xte = X[:n_train], X[n_train:]
    ytr, yte = y[:n_train], y[n_train:]

    y_ntk = kernel_regression(Xtr, ytr, Xte, lam=1e-2)
    y_mlp = wide_mlp_gd(Xtr, ytr, Xte, width=400, epochs=800, lr=0.05, seed=0)

    mse_ntk = float(np.mean((y_ntk - yte) ** 2))
    mse_mlp = float(np.mean((y_mlp - yte) ** 2))
    print(f"  Test MSE:")
    print(f"    NTK kernel regression       = {mse_ntk:.4f}")
    print(f"    Wide MLP (H=400, GD 800)    = {mse_mlp:.4f}")

    corr = float(np.corrcoef(y_ntk, y_mlp)[0, 1])
    print(f"  Correlation between NTK and MLP predictions on test: {corr:.4f}")
    print(f"  (Jacot 2018: as H -> inf, MLP-GD trajectories coincide with NTK regression.)")

    print("\n--- library cross-check (neural-tangents (Google) Python) ---")

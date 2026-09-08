"""Support vector regression (SVR) (Reference Sec 5.17).

Drucker et al. 1996; Smola & Scholkopf 2004. Regression analogue of
SVM: fits an epsilon-insensitive loss

    L_eps(y, f(x)) = max(0, |y - f(x)| - eps)

with regularisation ||f||^2 / 2. The DUAL problem introduces support
vectors + kernel k(x, x'):

    f(x) = sum_i (alpha_i - alpha_i^*) k(x_i, x) + b

Only points with |y - f(x)| >= eps have nonzero alpha (support
vectors). Common kernels: linear, RBF, polynomial.

We implement a compact quadratic-program-free SVR by iterating
coordinate ascent on the SMO-style dual (very small demo).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rbf_kernel(X, Y, gamma):
    if Y is None: Y = X
    D2 = (X[:, None, :] - Y[None, :, :]) ** 2
    return np.exp(-gamma * D2.sum(axis=-1))


def svr_fit_solve(X, y, C=1.0, eps=0.1, gamma=0.5):
    """Solve the dual by casting as a bounded-least-squares (dense small demo).

    minimises 0.5 alpha' K alpha + eps ||alpha||_1 - alpha' y   with
    -C <= alpha_i <= C.
    Uses a projected-gradient descent for a compact demo (not for large n).
    """
    n = len(X)
    K = rbf_kernel(X, X, gamma)
    alpha = np.zeros(n)
    lr = 1e-3
    for _ in range(3000):
        grad = K @ alpha - y + eps * np.sign(alpha)
        alpha = alpha - lr * grad
        alpha = np.clip(alpha, -C, C)
    b = float(np.mean(y - K @ alpha))
    return {"alpha": alpha, "b": b, "X_train": X, "gamma": gamma}


def svr_predict(model, X_new):
    K = rbf_kernel(X_new, model["X_train"], model["gamma"])
    return K @ model["alpha"] + model["b"]


if __name__ == "__main__":
    print("=== Support vector regression (SVR, epsilon-insensitive) ===\n")
    rng = np.random.default_rng(0)
    n = 100
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(X[:, 0]) + 0.2 * rng.normal(size=n)

    model = svr_fit_solve(X, y, C=1.0, eps=0.05, gamma=0.7)
    y_hat = svr_predict(model, X)
    mse = float(np.mean((y_hat - y) ** 2))
    n_sv = int(np.sum(np.abs(model["alpha"]) > 1e-3))
    print(f"  n = {n}   in-sample MSE = {mse:.4f}   support vectors = {n_sv}")

    #  Test on grid
    x_grid = np.linspace(-3, 3, 8).reshape(-1, 1)
    y_grid = svr_predict(model, x_grid)
    print(f"\n  Predictions on grid (should track sin(x)):")
    for x, yhat in zip(x_grid.flatten(), y_grid):
        print(f"    x = {x:+.2f}   SVR y = {yhat:+.3f}   sin(x) = {np.sin(x):+.3f}")

    print("\n--- library cross-check (e1071::svm(type='eps-regression') R;\n"
          "                          sklearn.svm.SVR Python) ---")

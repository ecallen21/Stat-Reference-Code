"""Thin-Plate Splines (Reference Sec 47.328).

Duchon 1977; Wahba 1990 SIAM. Minimum-curvature smoothing of
2-D scattered data:

    min sum (f(x_i) - y_i)^2 + lambda * ∫ ((f_xx)^2 + 2(f_xy)^2 + (f_yy)^2) dx dy

The optimum has the form

    f(x) = alpha_0 + alpha_1 x_1 + alpha_2 x_2 + sum w_i K(||x - x_i||)
    K(r) = r^2 log(r)

Closed-form solution via a block linear system. Standard for
2-D interpolation and geostatistics (kriging with linear drift).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def tps_kernel(r):
    r = np.maximum(r, 1e-9)
    return r ** 2 * np.log(r)


def fit_tps(x, y, values, lam=0.0):
    """2-D TPS fit; returns (weights, poly_coefs) usable for prediction."""
    n = len(x)
    pts = np.column_stack([x, y])
    D = np.sqrt(((pts[:, None] - pts[None, :]) ** 2).sum(axis=-1))
    K = tps_kernel(D)
    # Augment for polynomial part: [K + lam*I  P; P^T  0] [w; a] = [values; 0]
    P = np.column_stack([np.ones(n), x, y])
    A = np.block([[K + lam * np.eye(n), P],
                    [P.T, np.zeros((3, 3))]])
    b = np.concatenate([values, np.zeros(3)])
    sol = np.linalg.solve(A, b)
    w = sol[:n]; a = sol[n:]
    return w, a, pts


def predict_tps(x_new, y_new, w, a, pts):
    """Evaluate TPS at new points."""
    xn = np.column_stack([x_new, y_new])
    D = np.sqrt(((xn[:, None] - pts[None, :]) ** 2).sum(axis=-1))
    K = tps_kernel(D)
    return K @ w + a[0] + a[1] * x_new + a[2] * y_new


if __name__ == "__main__":
    print("=== Thin-Plate Splines (Duchon 1977; Wahba 1990) ===\n")
    rng = np.random.default_rng(0)

    # Truth: z = sin(2*pi*x) + 0.5*cos(4*pi*y)
    N = 60
    x = rng.uniform(0, 1, N); y = rng.uniform(0, 1, N)
    z = np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * y) + rng.normal(0, 0.05, N)

    w, a, pts = fit_tps(x, y, z, lam=0.01)
    print(f"  N = {N} scattered points, lambda (smoothing) = 0.01\n")

    # Test on a grid
    xg = np.linspace(0, 1, 10); yg = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(xg, yg)
    z_pred = predict_tps(xx.ravel(), yy.ravel(), w, a, pts).reshape(xx.shape)
    z_true = np.sin(2 * np.pi * xx) + 0.5 * np.cos(4 * np.pi * yy)

    rmse = float(np.sqrt(np.mean((z_pred - z_true) ** 2)))
    print(f"  Grid RMSE vs truth: {rmse:.3f}\n")

    print(f"  Predictions at four corners:")
    for xi, yi in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        p = float(predict_tps(np.array([xi]), np.array([yi]), w, a, pts)[0])
        t = float(np.sin(2 * np.pi * xi) + 0.5 * np.cos(4 * np.pi * yi))
        print(f"    ({xi}, {yi})   pred = {p:>+.3f}   true = {t:>+.3f}")

    print(f"\n  TPS uses a global radial-basis kernel r^2 log(r) - unlike Gaussian")
    print(f"  RBF, it grows unboundedly, capturing smooth global trend well.")

    print("\n--- library cross-check (scipy.interpolate.Rbf(function='thin_plate'); fields::Tps R) ---")

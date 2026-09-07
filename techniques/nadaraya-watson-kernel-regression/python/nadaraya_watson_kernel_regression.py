"""Nadaraya-Watson kernel regression (Reference Sec 5.15).

Nadaraya 1964; Watson 1964. Nonparametric regression estimator that
smooths local averages weighted by a kernel:

    m_hat(x) = sum_i K((x - x_i) / h) * y_i  /  sum_i K((x - x_i) / h)

with h the BANDWIDTH and K a kernel (Gaussian, Epanechnikov, uniform).

Bias-variance:
    * h small -> low bias, high variance (wiggly fit).
    * h large -> high bias, low variance (over-smoothed).

Rule-of-thumb bandwidth (Silverman): h = 1.06 * sigma_x * n^{-1/5}.
CV bandwidth selection is standard for actual work.

We fit N-W on a non-linear toy signal, compare a few bandwidths,
and cross-validate h.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gaussian_kernel(u):
    return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u):
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)


def nadaraya_watson(x, y, x_new, h, kernel=gaussian_kernel):
    """Point estimates at x_new."""
    x = np.asarray(x); y = np.asarray(y); x_new = np.asarray(x_new)
    m_hat = np.zeros(len(x_new))
    for i, x0 in enumerate(x_new):
        w = kernel((x - x0) / h)
        s = w.sum()
        m_hat[i] = (w * y).sum() / s if s > 0 else np.nan
    return m_hat


def leave_one_out_mse(x, y, h, kernel=gaussian_kernel):
    n = len(x)
    err = 0.0
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        w = kernel((x[mask] - x[i]) / h)
        s = w.sum()
        m_i = (w * y[mask]).sum() / s if s > 0 else 0.0
        err += (y[i] - m_i) ** 2
    return err / n


def cv_bandwidth(x, y, h_grid, kernel=gaussian_kernel):
    mses = [leave_one_out_mse(x, y, h, kernel) for h in h_grid]
    best_h = h_grid[int(np.argmin(mses))]
    return best_h, mses


if __name__ == "__main__":
    print("=== Nadaraya-Watson kernel regression ===\n")
    rng = np.random.default_rng(0)
    n = 200
    x = rng.uniform(-3, 3, size=n)
    y_true = np.sin(1.5 * x) + 0.3 * x
    y = y_true + rng.normal(scale=0.4, size=n)

    x_new = np.linspace(-3, 3, 60)
    #  Silverman rule of thumb bandwidth
    sigma_x = np.std(x, ddof=1)
    h_silver = 1.06 * sigma_x * n ** (-1 / 5)
    print(f"  n = {n}   sigma_x = {sigma_x:.3f}   Silverman h = {h_silver:.3f}\n")

    for h in [0.05, 0.2, h_silver, 1.0]:
        m_hat = nadaraya_watson(x, y, x_new, h)
        #  Compare to truth at those points
        m_true = np.sin(1.5 * x_new) + 0.3 * x_new
        rmse = float(np.sqrt(np.mean((m_hat - m_true) ** 2)))
        print(f"    h = {h:.3f}   RMSE(m_hat, m_true) = {rmse:.3f}")

    #  Cross-validate the bandwidth
    grid = np.linspace(0.05, 1.5, 15)
    best_h, mses = cv_bandwidth(x, y, grid)
    print(f"\n  CV-optimal bandwidth = {best_h:.3f}   (LOO-MSE = {min(mses):.3f})")
    m_hat_cv = nadaraya_watson(x, y, x_new, best_h)
    rmse_cv = float(np.sqrt(np.mean((m_hat_cv - (np.sin(1.5 * x_new) + 0.3 * x_new)) ** 2)))
    print(f"  RMSE(m_hat_CV, m_true) = {rmse_cv:.3f}")

    print("\n--- library cross-check (KernSmooth / stats::ksmooth R;\n"
          "                          statsmodels.nonparametric.KernelReg Python) ---")

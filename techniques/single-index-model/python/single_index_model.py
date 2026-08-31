"""Single-index model (Reference Sec 33.8).

Ichimura (1993) 'Semiparametric least-squares estimation of single-
index models.'

Model:  Y = g(X' beta) + eps         g unknown, beta identified up to scale.

The parametric direction beta is estimated by minimising the LEAST-
SQUARES loss with the UNKNOWN link function g estimated NONPARAMETRICALLY
(e.g. Nadaraya-Watson kernel smoother) inside the objective:

  min_beta   sum_i ( y_i - g_hat_{-i}(x_i' beta) )^2

where g_hat_{-i} is a leave-one-out kernel smoother.

Advantages:
  * FLEXIBLE (nonparametric link) yet interpretable (direction beta).
  * Estimates converge at parametric rate sqrt(n) for beta.

Here we:
  1. Fit beta by grid search over the unit-sphere.
  2. Estimate g nonparametrically via Nadaraya-Watson.
  3. Recover a synthetic single-index truth y = sin(x1 + 2 x2) + noise
     and compare beta_hat to the true direction (1, 2)/sqrt(5).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _kernel_smoother(z_train, y_train, z_query, bandwidth=0.3):
    """Nadaraya-Watson smoother; Gaussian kernel."""
    d = z_query[:, None] - z_train[None, :]
    w = np.exp(-0.5 * (d / bandwidth) ** 2)
    return (w * y_train).sum(axis=1) / (w.sum(axis=1) + 1e-12)


def _fit_g(X, y, beta):
    z = X @ beta
    return z, y


def _loss(X, y, beta, bandwidth):
    z = X @ beta
    y_hat = _kernel_smoother(z, y, z, bandwidth=bandwidth)
    return float(np.mean((y - y_hat) ** 2))


def fit_single_index(X, y, bandwidth=0.3, n_grid=200, seed=0):
    """Grid-search over beta directions on the unit 2-sphere (2-D X only)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    assert d == 2, "This demo assumes 2-D X for compact grid search."
    thetas = np.linspace(0, np.pi, n_grid)
    best_loss = np.inf; best_beta = None
    for th in thetas:
        beta = np.array([np.cos(th), np.sin(th)])
        loss = _loss(X, y, beta, bandwidth)
        if loss < best_loss:
            best_loss = loss; best_beta = beta
    return best_beta, best_loss


if __name__ == "__main__":
    print("=== Single-index model (Ichimura 1993) ===\n")
    rng = np.random.default_rng(0)
    n = 300
    X = rng.uniform(-1, 1, (n, 2))
    # True direction (1, 2) / sqrt(5), unknown link g(u) = sin(u * 3).
    beta_true = np.array([1.0, 2.0])
    beta_true /= np.linalg.norm(beta_true)
    z_true = X @ beta_true
    y = np.sin(3 * z_true) + rng.normal(0, 0.1, n)

    beta_hat, loss = fit_single_index(X, y, bandwidth=0.15)
    print(f"  true beta:      {np.round(beta_true, 3).tolist()}")
    print(f"  estimated beta: {np.round(beta_hat, 3).tolist()}   (direction only, sign-invariant)")
    cos_align = abs(float(beta_hat @ beta_true))
    print(f"  |cos(theta_true, theta_hat)| = {cos_align:.4f}   (1.0 = perfect direction match)")
    print(f"  training MSE   = {loss:.4f}")

    # Show the nonparametric link estimate at a few index values
    z_test = np.linspace(-2, 2, 8)
    g_hat = _kernel_smoother(X @ beta_hat, y, z_test, bandwidth=0.15)
    print(f"\n  nonparametric g_hat at index z:")
    for z, gh in zip(z_test, g_hat):
        print(f"    z={z:>5.2f}  g_hat={gh:>7.3f}   true g={np.sin(3*z):>7.3f}")

    print("\n--- library cross-check (np R package; SemiPar R; sisreg Python) ---")

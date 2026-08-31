"""Distributional regression (Reference Sec 33.12).

Klein-Kneib-Klasen-Lang (2015), Schmid-Wickler (2017), Rugamer-Kolb-Klein
(2023).

Predict the WHOLE conditional distribution F(y | x), not just the mean.
Three practical families:

  1. PARAMETRIC (GAMLSS): each distribution parameter is a regression
     (see gamlss).
  2. QUANTILE-BASED: fit many quantiles and stitch (see additive-
     quantile-regression).
  3. CDF/DENSITY-BASED: predict F(y | x) directly via a neural network
     or normalising flow (NGBoost, Duan 2020).

Here we implement a lightweight CDF-BASED distributional regressor:
  * Discretise y into K bins.
  * Fit a MULTINOMIAL softmax regression P(bin | x).
  * Report the predicted PMF, plus derived quantiles / prediction intervals
    per input.

Compare against a Gaussian-mean regression's fixed-variance prediction
interval.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_multinomial(X, y_bin, K, lr=0.3, epochs=800, l2=1e-3):
    d = X.shape[1]
    W = np.zeros((d, K))
    n = X.shape[0]
    Y = np.eye(K)[y_bin]
    for _ in range(epochs):
        p = _softmax(X @ W)
        g = X.T @ (p - Y) / n + l2 * W
        W -= lr * g
    return W


def predict_cdf(W, X, bin_edges):
    p = _softmax(X @ W)                              # (n, K)
    cdf = np.cumsum(p, axis=1)
    return p, cdf


def quantile_from_pmf(p, bin_centres, q):
    """Compute q-quantile per row from the PMF."""
    cdf = np.cumsum(p, axis=1)
    out = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        k = int(np.searchsorted(cdf[i], q))
        k = min(k, len(bin_centres) - 1)
        out[i] = bin_centres[k]
    return out


if __name__ == "__main__":
    print("=== Distributional regression (multinomial-bin CDF model) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    x = np.linspace(-2, 2, n)
    # Heteroscedastic + skewed: mean grows with x, variance grows with x, right skew for large x.
    y = (1.0 + 0.5 * x
          + (0.4 + 0.6 * np.abs(x)) * rng.normal(0, 1, n)
          + 0.3 * (x > 0.5) * rng.exponential(1, n))

    X = np.stack([np.ones(n), x], axis=1)
    K = 20
    bin_edges = np.linspace(y.min() - 0.1, y.max() + 0.1, K + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    y_bin = np.clip(np.searchsorted(bin_edges, y) - 1, 0, K - 1)

    W = fit_multinomial(X, y_bin, K, lr=0.3, epochs=600)
    p_hat, cdf = predict_cdf(W, X, bin_edges)

    # Report full quantile bands at three x-values.
    print(f"  {'x':>5}  {'q10':>7}  {'q50':>7}  {'q90':>7}   truth (mean, sd)")
    for xv in (-1.5, 0.0, 1.5):
        row = np.array([[1.0, xv]])
        p_v, _ = predict_cdf(W, row, bin_edges)
        q10 = quantile_from_pmf(p_v, bin_centres, 0.10)[0]
        q50 = quantile_from_pmf(p_v, bin_centres, 0.50)[0]
        q90 = quantile_from_pmf(p_v, bin_centres, 0.90)[0]
        true_mean = 1.0 + 0.5 * xv
        true_sd = float(0.4 + 0.6 * abs(xv))
        print(f"  {xv:>5.1f}  {q10:>7.3f}  {q50:>7.3f}  {q90:>7.3f}"
              f"   ({true_mean:.2f}, {true_sd:.2f})")

    print("\n  Compare to an OLS Gaussian mean model with fixed sd (won't spread as x grows).\n")
    print("--- library cross-check (ngboost; distributional-forests DRF; ML gluon TS) ---")

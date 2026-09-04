"""Yeo-Johnson transformation (Reference Sec 41.2).

Yeo & Johnson (2000) generalise Box-Cox to any real y (positive,
zero, negative):

    y >= 0 :  ((y + 1)^lambda - 1) / lambda      if lambda != 0
              log(y + 1)                          if lambda == 0
    y  < 0 : -((-y + 1)^(2 - lambda) - 1) / (2 - lambda)  if lambda != 2
              -log(-y + 1)                        if lambda == 2

Choose lambda by MLE (profile log-likelihood) as in Box-Cox.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def yeo_johnson(y, lam):
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    pos = y >= 0
    if abs(lam) < 1e-8:
        out[pos] = np.log(y[pos] + 1)
    else:
        out[pos] = ((y[pos] + 1) ** lam - 1) / lam
    if abs(lam - 2) < 1e-8:
        out[~pos] = -np.log(-y[~pos] + 1)
    else:
        out[~pos] = -((-y[~pos] + 1) ** (2 - lam) - 1) / (2 - lam)
    return out


def yj_loglik(y, lam):
    n = len(y); yt = yeo_johnson(y, lam)
    var = yt.var(ddof=1)
    ll = -0.5 * n * np.log(var) + (lam - 1) * np.sign(y).dot(np.log(np.abs(y) + 1))
    return float(ll)


def yj_mle(y, lambdas=None):
    if lambdas is None:
        lambdas = np.linspace(-3, 3, 301)
    ll = np.array([yj_loglik(y, lam) for lam in lambdas])
    idx = int(np.argmax(ll))
    return {"lambda_hat": float(lambdas[idx]), "loglik": float(ll[idx])}


if __name__ == "__main__":
    print("=== Yeo-Johnson transformation: handles zero and negative values ===\n")
    rng = np.random.default_rng(0)
    n = 400
    # Skewed data including zeros and negatives (Yeo-Johnson strength)
    y = rng.lognormal(mean=0, sigma=0.9, size=n) - 2.0

    res = yj_mle(y)
    print(f"  MLE lambda_YJ = {res['lambda_hat']:+.3f}   (log-lik = {res['loglik']:.2f})")
    _, p_raw = stats.shapiro(y)
    y_t = yeo_johnson(y, res["lambda_hat"])
    _, p_yj = stats.shapiro(y_t)
    print(f"  Shapiro p raw = {p_raw:.2e}   after YJ = {p_yj:.2e}")
    print(f"  skewness raw = {stats.skew(y):+.3f}   after YJ = {stats.skew(y_t):+.3f}\n")

    print("--- library cross-check (R car::powerTransform family='yjPower'; Python sklearn PowerTransformer) ---")

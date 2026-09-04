"""Box-Cox transformation (Reference Sec 41.1).

Box & Cox (1964) power transformation:

    y(lambda) = (y^lambda - 1) / lambda     if lambda != 0
              = log(y)                       if lambda == 0

Choose lambda by MLE (profile log-likelihood) to induce approximate
normality / homoscedasticity of residuals.  Special cases:
  lambda = -1     reciprocal
  lambda =  0     log
  lambda =  0.5   square root
  lambda =  1     no transformation
  lambda =  2     square

Requires strictly positive y; for y with zeros or negatives use
Yeo-Johnson (see companion technique).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def box_cox(y, lam):
    y = np.asarray(y, dtype=float)
    if abs(lam) < 1e-8:
        return np.log(y)
    return (y ** lam - 1) / lam


def box_cox_loglik(y, lam):
    """Profile log-likelihood for lambda (normal errors assumed)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    yt = box_cox(y, lam)
    var = yt.var(ddof=1)
    ll = -0.5 * n * np.log(var) + (lam - 1) * np.log(y).sum()
    return float(ll)


def box_cox_mle(y, lambdas=None):
    if lambdas is None:
        lambdas = np.linspace(-2, 2, 201)
    ll = np.array([box_cox_loglik(y, lam) for lam in lambdas])
    idx = int(np.argmax(ll))
    return {"lambda_hat": float(lambdas[idx]), "loglik": float(ll[idx]),
            "loglik_curve": list(zip(lambdas.tolist(), ll.tolist()))}


if __name__ == "__main__":
    print("=== Box-Cox transformation: MLE lambda + normality check ===\n")
    rng = np.random.default_rng(0)
    n = 400
    # Right-skewed positive y; ideal lambda ~ 0 (log)
    y = rng.lognormal(mean=1.0, sigma=0.6, size=n)

    res = box_cox_mle(y)
    print(f"  n = {n}   MLE lambda = {res['lambda_hat']:+.3f}"
          f"   (log-lik = {res['loglik']:.2f})")

    from scipy import stats
    _, p_raw = stats.shapiro(y)
    y_t = box_cox(y, res["lambda_hat"])
    _, p_bc = stats.shapiro(y_t)
    print(f"\n  Shapiro-Wilk normality p:")
    print(f"    raw y      = {p_raw:.2e}")
    print(f"    y(lambda)  = {p_bc:.2e}")
    print(f"    skewness raw = {stats.skew(y):+.3f}   after Box-Cox = {stats.skew(y_t):+.3f}\n")
    print("--- library cross-check (R MASS::boxcox; Python scipy.stats.boxcox / sklearn.PowerTransformer) ---")

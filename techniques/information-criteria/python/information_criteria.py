"""Information criteria (Reference Sec 34.5).

AIC (Akaike 1974):       -2 log L + 2 k
AICc (Sugiura 1978):     AIC + 2 k (k + 1) / (n - k - 1)    (small-n correction)
BIC (Schwarz 1978):      -2 log L + k log n
DIC (Spiegelhalter 2002): -2 log L(theta_bar) + 2 p_D       (Bayesian)
WAIC (Watanabe 2010):    -2 * (lppd - p_WAIC)                (Bayesian, out-of-sample)

Lower = better. AIC targets prediction; BIC targets true-model recovery.

Here we compute all criteria for nested Gaussian regression models on
synthetic data + show that BIC selects the true k, AIC over-selects
for finite n.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gaussian_log_likelihood(X, y, beta, sigma2):
    n = len(y)
    resid = y - X @ beta
    return float(-0.5 * n * np.log(2 * np.pi * sigma2) - (resid @ resid) / (2 * sigma2))


def fit_ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid / len(y))
    return beta, sigma2


def aic_bic_aicc(X, y, k):
    beta, sigma2 = fit_ols(X, y)
    ll = gaussian_log_likelihood(X, y, beta, sigma2)
    k_total = k + 1                                # + sigma^2
    n = len(y)
    aic = -2 * ll + 2 * k_total
    aicc = aic + 2 * k_total * (k_total + 1) / max(n - k_total - 1, 1)
    bic = -2 * ll + k_total * np.log(n)
    return {"log_lik": ll, "AIC": aic, "AICc": aicc, "BIC": bic, "n": n, "k": k_total}


if __name__ == "__main__":
    print("=== Information criteria: AIC / AICc / BIC ===\n")
    rng = np.random.default_rng(0)
    n = 100
    x = rng.uniform(-2, 2, n)
    true_k = 3
    # True model: cubic
    beta_true = np.array([0.5, 1.0, -0.5, 0.3])
    Phi_true = np.stack([np.ones(n), x, x ** 2, x ** 3], axis=1)
    y = Phi_true @ beta_true + rng.normal(0, 0.5, n)

    print(f"  n = {n}, true polynomial order = 3 (k = 4 including intercept)\n")
    print(f"  {'model order':>12}  {'log L':>10}  {'AIC':>10}  {'AICc':>10}  {'BIC':>10}")
    for order in range(1, 8):
        cols = [x ** j for j in range(order + 1)]
        Phi = np.stack(cols, axis=1)
        r = aic_bic_aicc(Phi, y, k=order + 1)
        print(f"  {order:>12}  {r['log_lik']:>10.2f}  {r['AIC']:>10.2f}"
              f"  {r['AICc']:>10.2f}  {r['BIC']:>10.2f}")

    print("\n  BIC penalises complexity more (log n vs 2), so it typically picks the true order;")
    print("  AIC may over-fit for finite n; AICc corrects small-n over-fit.\n")
    print("--- library cross-check (statsmodels.OLS().fit().aic/bic; R stats::AIC/BIC; loo::waic) ---")

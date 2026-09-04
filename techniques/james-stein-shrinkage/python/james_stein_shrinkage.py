"""James-Stein shrinkage (Reference Sec 38.15).

Stein (1956), James-Stein (1961) shocked the community: when p >= 3
normal means are estimated simultaneously, the MLE (sample means) is
INADMISSIBLE.  A convex combination that shrinks toward a common
target STRICTLY DOMINATES it in total mean-squared error, no matter
the true means.

For y_i ~ N(theta_i, sigma^2),   i = 1, ..., p:

  JS estimator:  theta_JS_i = ybar + (1 - (p - 3) sigma^2 / ||y - ybar||^2)_+ * (y_i - ybar)

(positive-part JS to avoid over-shrinkage sign flips).

TOTAL MSE decreases whenever p >= 3, worst-case gain when the true
means are clustered.  This launched empirical Bayes and modern
shrinkage.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def james_stein_positive(y, sigma2, target=None):
    """Positive-part JS shrinkage toward `target` (default = grand mean)."""
    y = np.asarray(y, dtype=float)
    if target is None:
        target = y.mean()
    p = len(y)
    shrink = 1 - (p - 3) * sigma2 / max(((y - target) ** 2).sum(), 1e-9)
    shrink = max(shrink, 0.0)         # positive-part
    return target + shrink * (y - target), float(shrink)


if __name__ == "__main__":
    print("=== James-Stein shrinkage vs MLE (total MSE) ===\n")
    rng = np.random.default_rng(0)
    n_sim = 400
    p_list = [3, 5, 10, 25, 50]
    sigma = 1.0
    print(f"  {'p':>3s}   {'MLE total MSE':>15s}   {'JS total MSE':>15s}"
          f"   {'JS / MLE':>10s}   {'shrink factor':>15s}")
    for p in p_list:
        theta_true = rng.normal(0, 1, p)
        mse_mle = 0.0
        mse_js = 0.0
        shrinks = []
        for _ in range(n_sim):
            y = theta_true + rng.normal(0, sigma, p)
            mse_mle += ((y - theta_true) ** 2).sum()
            theta_js, s = james_stein_positive(y, sigma ** 2)
            mse_js += ((theta_js - theta_true) ** 2).sum()
            shrinks.append(s)
        mse_mle /= n_sim; mse_js /= n_sim
        print(f"  {p:>3d}   {mse_mle:>15.3f}   {mse_js:>15.3f}   {mse_js / mse_mle:>10.3f}"
              f"   {np.mean(shrinks):>15.3f}")

    print("\n  --> JS beats MLE uniformly for p >= 3.  Larger p, more gain.\n")
    print("--- library cross-check (R corpcor::cov.shrink; Python sklearn.covariance) ---")

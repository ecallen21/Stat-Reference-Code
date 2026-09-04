"""Prediction vs confidence intervals (Reference Sec 39.14).

CONFIDENCE INTERVAL (CI) for the mean response at x*:
    xbar +/- t * s * sqrt(1/n + (x* - xbar)^2 / Sxx)

PREDICTION INTERVAL (PI) for a NEW individual observation at x*:
    xbar +/- t * s * sqrt(1 + 1/n + (x* - xbar)^2 / Sxx)

The PI is always WIDER because it must cover residual noise.  In
clinical practice this is critical: a lab-value regression predicts
where the group mean lies (CI) vs where a new patient will lie (PI).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # t-quantile


def linreg_intervals(x, y, x_new, alpha=0.05):
    """Fit y ~ x, return CI + PI at x_new."""
    n = len(x)
    xbar = x.mean(); ybar = y.mean()
    Sxx = ((x - xbar) ** 2).sum()
    beta1 = ((x - xbar) * (y - ybar)).sum() / Sxx
    beta0 = ybar - beta1 * xbar
    yhat = beta0 + beta1 * x
    s = np.sqrt(((y - yhat) ** 2).sum() / (n - 2))       # residual SD
    t = stats.t.ppf(1 - alpha / 2, df=n - 2)
    m = beta0 + beta1 * x_new
    se_ci = s * np.sqrt(1 / n + (x_new - xbar) ** 2 / Sxx)
    se_pi = s * np.sqrt(1 + 1 / n + (x_new - xbar) ** 2 / Sxx)
    return {"x_new": x_new, "y_hat": m, "ci": (m - t * se_ci, m + t * se_ci),
            "pi": (m - t * se_pi, m + t * se_pi), "ci_width": 2 * t * se_ci,
            "pi_width": 2 * t * se_pi, "sigma_hat": float(s)}


if __name__ == "__main__":
    print("=== Prediction interval vs confidence interval (linear model) ===\n")
    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(50, 90, n)                # age
    y = 100 + 0.7 * x + rng.normal(0, 12, n)    # SBP-like

    for x_new in (55, 70, 85, 95):
        r = linreg_intervals(x, y, x_new, alpha=0.05)
        print(f"  x* = {x_new:3d}   yhat = {r['y_hat']:.2f}"
              f"   CI = [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  (w = {r['ci_width']:.2f})"
              f"   PI = [{r['pi'][0]:.2f}, {r['pi'][1]:.2f}]  (w = {r['pi_width']:.2f})")

    # Coverage sanity check: PI should cover about 95% of new observations
    B = 3000
    covers_ci = 0; covers_pi = 0
    x0 = 70.0
    for _ in range(B):
        x_sim = rng.uniform(50, 90, n)
        y_sim = 100 + 0.7 * x_sim + rng.normal(0, 12, n)
        r = linreg_intervals(x_sim, y_sim, x0, alpha=0.05)
        # True mean at x0 = 100 + 0.7*x0
        mu_true = 100 + 0.7 * x0
        covers_ci += r["ci"][0] <= mu_true <= r["ci"][1]
        # A new observation at x0:
        y_new = mu_true + rng.normal(0, 12)
        covers_pi += r["pi"][0] <= y_new <= r["pi"][1]

    print(f"\n  Simulated coverage @ x* = {x0}   CI covers true mean: {covers_ci / B:.3f}"
          f"    PI covers new obs: {covers_pi / B:.3f}  (target 0.95)\n")
    print("--- library cross-check (R stats::predict(interval='prediction'); Python statsmodels wls_prediction_std) ---")

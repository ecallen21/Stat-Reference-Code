"""Meta-analysis: fixed + random effects + heterogeneity (Reference §20.1).

Combine effect-size estimates y_i from k studies with within-study variances
v_i to produce a pooled estimate.

Fixed-effects (inverse-variance weighted)
    w_i = 1 / v_i
    y_bar = sum(w_i y_i) / sum(w_i)
    Var(y_bar) = 1 / sum(w_i)
Assumes ALL studies estimate the SAME true effect.

Random-effects (DerSimonian-Laird 1986)
    y_i ~ Normal(theta_i, v_i)
    theta_i ~ Normal(mu, tau^2)      between-study variance
    Combined weights: w_i^* = 1 / (v_i + tau_hat^2)
    Method-of-moments tau^2:
        Q = sum(w_i (y_i - y_bar)^2)
        tau_hat^2 = max(0, (Q - (k - 1)) / (sum(w_i) - sum(w_i^2) / sum(w_i)))

Heterogeneity
    I^2 = max(0, (Q - (k - 1)) / Q) * 100    % of variance due to between-study
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def fixed_effect(yi, vi, ci: float = 0.95) -> dict:
    yi = np.asarray(yi, dtype=float); vi = np.asarray(vi, dtype=float)
    w = 1 / vi
    y_bar = float(np.sum(w * yi) / np.sum(w))
    var_bar = float(1 / np.sum(w))
    se = math.sqrt(var_bar)
    q_z = stats.norm.ppf(1 - (1 - ci) / 2)
    return {"estimate": y_bar, "se": se,
            "ci_low": y_bar - q_z * se, "ci_high": y_bar + q_z * se,
            "z": y_bar / se, "p_value": float(2 * stats.norm.sf(abs(y_bar / se))),
            "k_studies": int(len(yi)),
            "method": "Fixed-effects meta-analysis (inverse-variance)"}


def random_effects_DL(yi, vi, ci: float = 0.95) -> dict:
    """DerSimonian-Laird random-effects meta-analysis."""
    yi = np.asarray(yi, dtype=float); vi = np.asarray(vi, dtype=float)
    k = len(yi)
    w = 1 / vi
    y_bar_fe = float(np.sum(w * yi) / np.sum(w))
    Q = float(np.sum(w * (yi - y_bar_fe) ** 2))
    df = k - 1
    numer = Q - df
    denom = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, numer / denom) if denom > 0 else 0.0
    w_re = 1 / (vi + tau2)
    y_bar = float(np.sum(w_re * yi) / np.sum(w_re))
    se = float(math.sqrt(1 / np.sum(w_re)))
    q_z = stats.norm.ppf(1 - (1 - ci) / 2)
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return {"estimate": y_bar, "se": se,
            "ci_low": y_bar - q_z * se, "ci_high": y_bar + q_z * se,
            "tau2": tau2, "Q": Q, "df": df,
            "Q_p_value": float(stats.chi2.sf(Q, df)),
            "I2_percent": I2,
            "k_studies": int(k),
            "method": "Random-effects meta-analysis (DerSimonian-Laird)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 8 studies with true log-OR = 0.35 and heterogeneity tau = 0.15
    k = 8
    theta = rng.normal(0.35, 0.15, k)
    vi = rng.uniform(0.03, 0.12, k)
    yi = rng.normal(theta, np.sqrt(vi))

    print(f"=== Meta-analysis of k = {k} studies (true mean effect ~ 0.35) ===")
    print("\n  study  y_i     v_i")
    for i, (y, v) in enumerate(zip(yi, vi)):
        print(f"  {i + 1:5d}  {y:6.3f}  {v:.4f}")

    print("\n=== Fixed-effect ===")
    r = fixed_effect(yi, vi)
    print(f"  pooled = {r['estimate']:.4f}   95% CI = ({r['ci_low']:.4f}, {r['ci_high']:.4f})")
    print(f"  z = {r['z']:.2f}, p = {r['p_value']:.4f}")

    print("\n=== Random-effects (DerSimonian-Laird) ===")
    r = random_effects_DL(yi, vi)
    print(f"  pooled = {r['estimate']:.4f}   95% CI = ({r['ci_low']:.4f}, {r['ci_high']:.4f})")
    print(f"  tau^2 = {r['tau2']:.4f}   Q = {r['Q']:.3f} (df {r['df']}, p = {r['Q_p_value']:.4f})")
    print(f"  I^2 = {r['I2_percent']:.1f}%")

    print("\n--- library cross-check (metafor in R) ---")
    print("  R: metafor::rma(yi = yi, vi = vi, method = 'DL')")

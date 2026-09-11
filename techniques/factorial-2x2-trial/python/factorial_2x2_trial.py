"""2x2 Factorial Trial (Reference Sec 47.267).

Piantadosi 2005 Clinical Trials: A Methodologic Perspective;
McAlister et al 2003 JAMA. Randomise TWO interventions
simultaneously in a 2x2 grid:

    outcome = mu + a * A + b * B + c * (A * B) + e

Analysis estimates:
    a = main effect of intervention A
    b = main effect of intervention B
    c = A*B interaction

Efficient - tests two interventions with (almost) the sample
size of a single-arm trial. BUT the presence of an interaction
inflates variance and can mask main effects.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def simulate_factorial(n_per_cell, mu, a, b, c, sigma, rng):
    """Simulate 2x2 factorial trial data."""
    rows = []
    for A in [0, 1]:
        for B in [0, 1]:
            for _ in range(n_per_cell):
                y = mu + a * A + b * B + c * A * B + rng.normal(0, sigma)
                rows.append((A, B, y))
    return np.array(rows, dtype=float)


def fit_factorial(data):
    """Fit y ~ A + B + A*B by OLS; return dict of coefs + SEs + p-values."""
    from scipy import stats
    A = data[:, 0]; B = data[:, 1]; y = data[:, 2]
    X = np.column_stack([np.ones(len(y)), A, B, A * B])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid ** 2).sum() / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return dict(coefs=beta, se=se, p=p, names=["intercept", "A", "B", "A*B"])


if __name__ == "__main__":
    print("=== 2x2 Factorial Trial (Piantadosi 2005) ===\n")
    rng = np.random.default_rng(0)

    print("  Scenario 1: NO interaction (c = 0)")
    print(f"  {'a':>4}  {'b':>4}  {'c':>4}  a_hat SE   b_hat SE   AB_hat SE   n=100/cell")
    data = simulate_factorial(100, mu=1.0, a=0.5, b=0.4, c=0.0, sigma=1.0, rng=rng)
    fit = fit_factorial(data)
    print(f"  {0.5:>4}  {0.4:>4}  {0.0:>4}  {fit['coefs'][1]:>5.2f} {fit['se'][1]:.2f}   "
          f"{fit['coefs'][2]:>5.2f} {fit['se'][2]:.2f}   "
          f"{fit['coefs'][3]:>5.2f} {fit['se'][3]:.2f}\n")

    print("  Scenario 2: WITH interaction (c = 0.6)")
    data = simulate_factorial(100, mu=1.0, a=0.5, b=0.4, c=0.6, sigma=1.0, rng=rng)
    fit = fit_factorial(data)
    for i, name in enumerate(fit["names"]):
        print(f"    {name:>10}   coef = {fit['coefs'][i]:>6.3f}   SE = {fit['se'][i]:.3f}   "
              f"p = {fit['p'][i]:.4f}")

    print(f"\n  Efficiency comparison: 2x2 factorial vs two separate trials")
    n_per_arm = 200
    # Single 2-arm trial for A: need 2*n patients
    # Factorial: 4 groups of n_per_cell = n_per_arm patients each => n_per_arm patients contribute to A vs not-A
    # main effect A has SE ~ 2 * sigma / sqrt(N) where N = 4 * n_per_cell
    # single trial:  SE ~ 2 * sigma / sqrt(2 * n_per_arm)
    N_fact = 4 * n_per_arm
    se_fact_A = 2 * 1.0 / np.sqrt(N_fact)
    se_two   = 2 * 1.0 / np.sqrt(2 * n_per_arm)
    print(f"    2x2 factorial (N_total = {N_fact}): SE(A) ~ {se_fact_A:.4f}")
    print(f"    Single trial   (N_total = {2 * n_per_arm}): SE(A) ~ {se_two:.4f}")
    print(f"    Factorial gains info on B for free; if interaction is small, this is efficient.")

    print("\n--- library cross-check (base R lm; statsmodels ols; anova() in both) ---")

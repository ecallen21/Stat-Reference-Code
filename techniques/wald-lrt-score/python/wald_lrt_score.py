"""The three classical likelihood-based tests: Wald, LRT, and Score (Rao).
(Reference §3.18, §3.30, §3.31, §3.33).

All three test H0: theta = theta_0 against H1: theta != theta_0 in a parametric
model with log-likelihood ell(theta). They are asymptotically equivalent under
H0 (each statistic -> chi^2_p), but differ in:

  - What you need to compute
  - Their finite-sample behavior
  - Invariance to reparameterization

Statistics
----------
  Wald:    W = (theta_hat - theta_0)^T I(theta_hat) (theta_hat - theta_0)
           uses the MLE and the observed Fisher information at the MLE.
           Equivalent: W = (theta_hat - theta_0) / SE(theta_hat)  (scalar form,
           then squared).
  LRT:     -2 log Lambda = -2 (ell(theta_0) - ell(theta_hat))
           need both the constrained ell(theta_0) and the unconstrained MLE.
  Score:   S = U(theta_0)^T I(theta_0)^{-1} U(theta_0)
           uses ONLY the null (no MLE needed); good when fitting at the MLE
           is hard.

Where U = d ell / d theta is the score function and I is the Fisher information.

For a regular smooth model with one parameter:
                                 evaluated at:
  Wald   = (theta_hat - theta_0)^2 * I(theta_hat)        | theta_hat
  LRT    = 2 * (ell(theta_hat) - ell(theta_0))           | both
  Score  = U(theta_0)^2 / I(theta_0)                     | theta_0

Demo: binomial(n, p) with n trials, x successes, H0: p = p_0.
  ell(p)   = x log p + (n - x) log (1 - p)
  ell'(p)  = x/p - (n - x)/(1 - p)             [score U(p)]
  ell''(p) = -x/p^2 - (n - x)/(1 - p)^2        [-information I(p)]
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint meaning 'a function'

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


# ---------- Generic 1-D implementations -----------------------------------
def wald_test(theta_hat: float, theta_0: float, information_at_hat: float) -> dict:
    """Wald test for a scalar parameter.

    Parameters
    ----------
    theta_hat : MLE.
    theta_0   : null value.
    information_at_hat : Fisher information evaluated at the MLE (1/SE^2).
    """
    se = math.sqrt(1 / information_at_hat)
    W = (theta_hat - theta_0) ** 2 * information_at_hat
    p = float(stats.chi2.sf(W, df=1))
    return {"statistic": W, "df": 1, "p_value": p, "se": se, "method": "Wald"}


def likelihood_ratio_test(loglik_hat: float, loglik_null: float, df: int = 1) -> dict:
    """LRT given the maximized log-likelihood and the constrained log-likelihood."""
    G2 = 2 * (loglik_hat - loglik_null)
    return {"statistic": G2, "df": df,
            "p_value": float(stats.chi2.sf(G2, df)),
            "method": "Likelihood Ratio"}


def score_test(score_at_null: float, information_at_null: float) -> dict:
    """Score (Rao / Lagrange-multiplier) test for a scalar parameter."""
    S = score_at_null ** 2 / information_at_null
    return {"statistic": S, "df": 1, "p_value": float(stats.chi2.sf(S, df=1)),
            "method": "Score (Rao)"}


# ---------- Worked example: binomial proportion ---------------------------
def binomial_three_tests(x: int, n: int, p0: float) -> dict:
    """All three tests of H0: p = p0 for x successes in n binomial trials."""
    p_hat = x / n
    # log-likelihoods
    def ll(p):
        # use 0 log 0 = 0 (the standard convention) so boundary cases work
        def xlogy(a, b): return 0.0 if a == 0 else a * math.log(b) if b > 0 else -math.inf
        return xlogy(x, p) + xlogy(n - x, 1 - p)
    ll_hat = ll(p_hat); ll_null = ll(p0)
    # information I(p) = n / (p (1 - p)) under Bernoulli
    I_hat = n / (p_hat * (1 - p_hat)) if 0 < p_hat < 1 else float("inf")
    I_null = n / (p0 * (1 - p0))
    # score U(p) = x/p - (n - x)/(1 - p)
    U_null = x / p0 - (n - x) / (1 - p0)

    return {
        "p_hat": p_hat,
        "wald": wald_test(p_hat, p0, I_hat),
        "lrt": likelihood_ratio_test(ll_hat, ll_null, df=1),
        "score": score_test(U_null, I_null),
    }


def library_versions(x: int, n: int, p0: float):
    # statsmodels provides the proportions_chisquare etc., but for a clean
    # cross-check we just compute the binomial log-likelihoods directly above.
    from statsmodels.stats.proportion import proportions_ztest
    z, p = proportions_ztest(x, n, p0, prop_var=p0)   # score-flavored z
    return {"statsmodels proportions_ztest (score-flavored)": (float(z) ** 2,
                                                                 float(p))}


if __name__ == "__main__":
    # Coin with x = 60 successes in n = 100; H0: p = 0.5
    res = binomial_three_tests(60, 100, 0.5)
    print(f"x = 60, n = 100, p0 = 0.5  (p_hat = {res['p_hat']})\n")
    for name in ("wald", "lrt", "score"):
        r = res[name]
        print(f"  {r['method']:18s}: statistic = {r['statistic']:.4f}  p = {r['p_value']:.4f}")

    # Subtle case where the three diverge: x = 0 (boundary)
    print("\nBoundary case  x = 0, n = 20, p0 = 0.1 :")
    res = binomial_three_tests(0, 20, 0.1)
    for name in ("wald", "lrt", "score"):
        r = res[name]
        print(f"  {r['method']:18s}: statistic = {r['statistic']:.4f}  p = {r['p_value']:.4f}")
    print("(Wald is unreliable near the boundary -- the LRT/score behave better.)")

    print("\n--- library (statsmodels score-z^2 for binomial) ---")
    for k, v in library_versions(60, 100, 0.5).items():
        print(f"  {k}: {v}")

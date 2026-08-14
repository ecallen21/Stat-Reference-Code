"""Bayesian A/B testing on binomial conversion rates (Reference §14.33).

Two Bernoulli arms A and B with unknown conversion rates p_A, p_B.
Beta priors + observed successes / trials -> posterior Betas.

Report FULL POSTERIORS rather than a single p-value:
    Pr(p_B > p_A | data)   direct probability of superiority
    Posterior distribution of the LIFT (p_B - p_A) or RATIO (p_B / p_A)
    Expected loss under a stopping rule

Contrast with frequentist A/B: no peeking penalty, easy to interpret,
naturally handles unequal sample sizes.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def bayesian_ab(y_A: int, n_A: int, y_B: int, n_B: int,
                prior_a: float = 1.0, prior_b: float = 1.0,
                n_mc: int = 100000, seed: int = 0) -> dict:
    """Beta-Binomial Bayesian A/B test with Monte Carlo posterior draws."""
    rng = np.random.default_rng(seed)
    a_A = prior_a + y_A; b_A = prior_b + n_A - y_A
    a_B = prior_a + y_B; b_B = prior_b + n_B - y_B
    p_A_draws = rng.beta(a_A, b_A, n_mc)
    p_B_draws = rng.beta(a_B, b_B, n_mc)
    lift = p_B_draws - p_A_draws
    ratio = p_B_draws / p_A_draws
    prob_B_beats_A = float((p_B_draws > p_A_draws).mean())
    # Expected loss (under the wrong choice, per Chris Stucchio's formulation)
    expected_loss_choose_B = float(np.mean(np.maximum(p_A_draws - p_B_draws, 0)))
    expected_loss_choose_A = float(np.mean(np.maximum(p_B_draws - p_A_draws, 0)))
    return {"posterior_alpha_A": float(a_A), "posterior_beta_A": float(b_A),
            "posterior_alpha_B": float(a_B), "posterior_beta_B": float(b_B),
            "posterior_mean_A": float(a_A / (a_A + b_A)),
            "posterior_mean_B": float(a_B / (a_B + b_B)),
            "lift_mean": float(lift.mean()),
            "lift_ci_95": (float(np.quantile(lift, 0.025)), float(np.quantile(lift, 0.975))),
            "ratio_mean": float(ratio.mean()),
            "P(B > A)": prob_B_beats_A,
            "expected_loss_choose_B": expected_loss_choose_B,
            "expected_loss_choose_A": expected_loss_choose_A,
            "method": "Bayesian A/B test (Beta-Binomial + Monte Carlo)"}


if __name__ == "__main__":
    print("=== Bayesian A/B test: 120/1000 vs 145/1000, Uniform prior ===")
    r = bayesian_ab(y_A=120, n_A=1000, y_B=145, n_B=1000)
    print(f"  posterior p_A = {r['posterior_mean_A']:.4f}   posterior p_B = {r['posterior_mean_B']:.4f}")
    print(f"  posterior lift mean = {r['lift_mean']:.4f}   95% CI = ({r['lift_ci_95'][0]:.4f}, {r['lift_ci_95'][1]:.4f})")
    print(f"  posterior ratio mean = {r['ratio_mean']:.3f}")
    print(f"  P(B > A) = {r['P(B > A)']:.4f}")
    print(f"  expected loss if you choose B = {r['expected_loss_choose_B']:.5f}")
    print(f"  expected loss if you choose A = {r['expected_loss_choose_A']:.5f}")

    print("\n=== Ties / small samples: 3/50 vs 5/50 ===")
    r = bayesian_ab(y_A=3, n_A=50, y_B=5, n_B=50)
    print(f"  posterior lift mean = {r['lift_mean']:.4f}   95% CI = ({r['lift_ci_95'][0]:.4f}, {r['lift_ci_95'][1]:.4f})")
    print(f"  P(B > A) = {r['P(B > A)']:.4f}")

    print("\n--- library cross-check (PyMC BayesianAB or bayesian-tests-book) ---")
    print("  See e.g. bayestestR::posterior_odds for the general Bayesian testing interface.")

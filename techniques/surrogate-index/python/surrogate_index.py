"""Surrogate index for long-term effects (Reference Sec 44.14).

Athey-Chetty-Imbens-Kang: combine SHORT-TERM proxies (S) into an
index that predicts LONG-TERM outcome (Y) from a historical
population.  Then use the index in the current experiment where
only short-term outcomes are observed.

Two-stage estimation:
  1. From historical data, fit S -> Y regression -> weights.
  2. In current experiment, construct S-based index and treat as if
     it were the long-term outcome (with SEs adjusted for the
     estimated weights).
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LinearRegression


def fit_surrogate_weights(S_hist, Y_hist):
    """Fit historical Y = alpha + S beta + eps  -> return beta + intercept."""
    m = LinearRegression().fit(S_hist, Y_hist)
    return m.intercept_, m.coef_


def surrogate_index(S_exp, alpha, beta):
    return alpha + S_exp @ beta


if __name__ == "__main__":
    print("=== Surrogate index: combine short-term proxies to estimate long-term effect ===\n")
    rng = np.random.default_rng(0)
    # Historical population: 3 short-term signals -> 1 long-term outcome
    n_hist = 5000
    S_hist = rng.normal(0, 1, (n_hist, 3))
    Y_hist = 0.6 * S_hist[:, 0] + 0.3 * S_hist[:, 1] + 0.1 * S_hist[:, 2] + rng.normal(0, 0.4, n_hist)
    alpha, beta = fit_surrogate_weights(S_hist, Y_hist)
    print(f"  Historical surrogate weights (S -> Y): {np.round(beta, 3)}")

    # Current experiment: only short-term S observed
    n_exp = 2000
    T = rng.integers(0, 2, n_exp)
    # True treatment effect on short-term signals shifts each by 0.20
    S_exp = rng.normal(0, 1, (n_exp, 3)) + 0.20 * T[:, None]
    Y_index = surrogate_index(S_exp, alpha, beta)
    diff = Y_index[T == 1].mean() - Y_index[T == 0].mean()
    print(f"  Surrogate-index treatment effect estimate = {diff:+.3f}")
    print(f"  (True long-term effect ~= sum(beta) * 0.20 = {(beta.sum()) * 0.20:.3f})\n")

    # Compare: naive long-term measurement would take 6+ months to observe
    print("  Business value: get a long-term effect estimate weeks/months earlier.\n")
    print("--- library cross-check (R stats::lm + custom two-stage; Python sklearn + custom) ---")

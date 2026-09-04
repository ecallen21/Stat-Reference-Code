"""Heterogeneous treatment effect + uplift modelling (Reference Sec 44.7).

Athey-Imbens 2016, Kunzel et al. 2019.  Estimate CATE:
    tau(x) = E[Y(1) - Y(0) | X = x]

Meta-learners:
  T-learner : fit mu1(x), mu0(x) separately -> tau(x) = mu1 - mu0.
  S-learner : fit single model on (X, T) -> tau(x) = mu(x, 1) - mu(x, 0).
  X-learner : combines predictions with propensity weighting.

Uplift = ranking users by predicted CATE to target treatment to
those most likely to benefit.  Qini curve / uplift score evaluate
ranking quality.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.ensemble import GradientBoostingRegressor


def t_learner(X, T, Y):
    m0 = GradientBoostingRegressor(random_state=0).fit(X[T == 0], Y[T == 0])
    m1 = GradientBoostingRegressor(random_state=0).fit(X[T == 1], Y[T == 1])
    return lambda x: m1.predict(x) - m0.predict(x)


def s_learner(X, T, Y):
    X_aug = np.column_stack([X, T])
    m = GradientBoostingRegressor(random_state=0).fit(X_aug, Y)
    def cate(x):
        x1 = np.column_stack([x, np.ones(len(x))])
        x0 = np.column_stack([x, np.zeros(len(x))])
        return m.predict(x1) - m.predict(x0)
    return cate


def qini_score(tau_hat, Y, T):
    """Qini score = area under uplift curve."""
    order = np.argsort(-tau_hat)          # highest predicted uplift first
    Y_o = Y[order]; T_o = T[order]
    n = len(Y)
    n1 = T_o.cumsum(); n0 = (1 - T_o).cumsum()
    # Cumulative gain in outcome relative to random
    gain = (T_o * Y_o).cumsum() - (1 - T_o) * Y_o[T_o == 0].mean() * np.arange(1, n + 1)
    # Simple area under curve normalised by n
    return float(gain.mean() / n)


if __name__ == "__main__":
    print("=== Heterogeneous treatment effect: T-learner vs S-learner ===\n")
    rng = np.random.default_rng(0)
    n, p = 3000, 5
    X = rng.normal(0, 1, (n, p))
    T = rng.integers(0, 2, n)
    # True CATE = 1.0 * X[:, 0] (treatment helps X0-positive users, hurts X0-negative)
    tau_true = 1.0 * X[:, 0]
    Y = 5 + 0.3 * X[:, 1] + tau_true * T + rng.normal(0, 0.5, n)

    cate_t = t_learner(X, T, Y)(X)
    cate_s = s_learner(X, T, Y)(X)
    print(f"  Correlation T-learner CATE with true tau: {np.corrcoef(cate_t, tau_true)[0, 1]:.3f}")
    print(f"  Correlation S-learner CATE with true tau: {np.corrcoef(cate_s, tau_true)[0, 1]:.3f}")

    qini_t = qini_score(cate_t, Y, T)
    qini_s = qini_score(cate_s, Y, T)
    qini_rand = qini_score(rng.normal(0, 1, n), Y, T)
    print(f"\n  Qini score  T-learner = {qini_t:.3f}")
    print(f"  Qini score  S-learner = {qini_s:.3f}")
    print(f"  Qini score  random    = {qini_rand:.3f}\n")
    print("--- library cross-check (R grf::causal_forest, uplift; Python causalml/econml) ---")

"""Functional ANOVA (Reference Sec 31.4).

Ramsay-Silverman FDA (2005), Ch 13.

Test whether K groups have equal MEAN CURVES.  Under H0: mu_1(t) = ... =
mu_K(t) for all t. Point-wise F-statistic:

  F(t) = (SS_between(t) / (K - 1))  /  (SS_within(t) / (n - K)).

Global test uses:
  * SUP F: F_sup = max_t F(t)  -> permutation p-value.
  * INTEGRATED F: F_int = integral F(t) dt.
  * L2-norm of between-group mean differences.

Permutation test: shuffle group labels B times; the empirical p-value
is the fraction of shuffled F_sup (or F_int) >= observed.

Here we implement pointwise F + permutation p on synthetic 3-group
curves with a hidden mean shift.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def pointwise_f(X, group):
    """Return F(t) pointwise across the T grid."""
    groups = np.unique(group)
    K = len(groups)
    n = X.shape[0]
    grand = X.mean(axis=0)
    ss_b = np.zeros(X.shape[1])
    ss_w = np.zeros(X.shape[1])
    for g in groups:
        m = group == g
        Xg = X[m]
        mean_g = Xg.mean(axis=0)
        ss_b += m.sum() * (mean_g - grand) ** 2
        ss_w += ((Xg - mean_g) ** 2).sum(axis=0)
    return (ss_b / (K - 1)) / (ss_w / (n - K) + 1e-12)


def permutation_p(X, group, stat="sup", B=500, seed=0):
    rng = np.random.default_rng(seed)
    F_obs = pointwise_f(X, group)
    obs_stat = float(F_obs.max()) if stat == "sup" else float(F_obs.mean())
    group_perm = group.copy()
    hits = 0
    for _ in range(B):
        rng.shuffle(group_perm)
        F_perm = pointwise_f(X, group_perm)
        s = float(F_perm.max()) if stat == "sup" else float(F_perm.mean())
        if s >= obs_stat:
            hits += 1
    return obs_stat, hits / B, F_obs


if __name__ == "__main__":
    print("=== Functional ANOVA + permutation p-value ===\n")
    rng = np.random.default_rng(0)
    T = 100
    t = np.linspace(0, 1, T)
    def make_group(mean_fn, n=40, sigma=0.5):
        return np.array([mean_fn(t) + sigma * rng.normal(0, 1, T) for _ in range(n)])

    # Scenario A: three groups with DIFFERENT mean curves.
    X_A = np.vstack([make_group(lambda t: 2 * np.sin(2 * np.pi * t)),
                       make_group(lambda t: 2 * np.sin(2 * np.pi * t) + 0.8),
                       make_group(lambda t: 2 * np.sin(2 * np.pi * t) - 0.8)])
    g_A = np.array([0] * 40 + [1] * 40 + [2] * 40)
    obs_A, p_A, F_A = permutation_p(X_A, g_A, stat="sup")
    print(f"  Groups DIFFER:   sup F(t) = {obs_A:.2f}   permutation p = {p_A:.3f}")

    # Scenario B: all groups from the SAME mean curve.
    X_B = np.vstack([make_group(lambda t: 2 * np.sin(2 * np.pi * t)) for _ in range(3)])
    g_B = np.array([0] * 40 + [1] * 40 + [2] * 40)
    obs_B, p_B, F_B = permutation_p(X_B, g_B, stat="sup")
    print(f"  Groups EQUAL:    sup F(t) = {obs_B:.2f}   permutation p = {p_B:.3f}\n")

    print("--- library cross-check (R fda::Fperm.fd; fda.usc::anova.RPm) ---")

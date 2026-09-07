"""Buhlmann credibility (Reference Sec 24.20).

Buhlmann 1967 'Experience rating and credibility', ASTIN Bulletin.
Classic actuarial framework: predict a group's future loss from a
WEIGHTED average of the group's own experience and the collective
mean:

    P_g = Z_g * X_bar_g + (1 - Z_g) * mu

    Z_g = n_g / (n_g + K)          (credibility factor)
    K   = E[Var(X | g)] / Var(E[X | g])  =  sigma^2_within / tau^2_between

Higher n_g -> more credibility to own experience. Lower within-group
variance vs between-group variance -> more credibility to group.

Equivalent to a random-intercept BLUP: exactly the linear Bayes
estimator when losses are Normal + Normal prior.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def buhlmann_estimator(group_ids, X):
    """Compute Buhlmann credibility per group.

    group_ids : array of ints (group per observation).
    X          : array of losses per observation.
    """
    G = int(group_ids.max()) + 1
    n_g = np.array([np.sum(group_ids == g) for g in range(G)])
    mean_g = np.array([X[group_ids == g].mean() for g in range(G)])
    var_g = np.array([X[group_ids == g].var(ddof=1) if n_g[g] > 1 else np.nan for g in range(G)])
    grand_mean = float(X.mean())

    #  Within-group variance sigma^2 = mean of group variances (weighted by n_g - 1)
    valid = ~np.isnan(var_g)
    sigma2 = float(np.sum((n_g[valid] - 1) * var_g[valid]) / np.sum(n_g[valid] - 1))
    #  Between-group variance tau^2 = ANOVA-style (mean squares)
    var_group_means = float(np.var(mean_g, ddof=1))
    tau2 = max(var_group_means - sigma2 / np.mean(n_g), 0.0)
    K = sigma2 / tau2 if tau2 > 0 else np.inf
    Z = n_g / (n_g + K)
    prediction = Z * mean_g + (1 - Z) * grand_mean
    return {"grand_mean": grand_mean, "sigma2": sigma2, "tau2": tau2, "K": float(K),
            "Z_by_group": Z, "prediction_by_group": prediction,
            "raw_group_mean": mean_g, "n_by_group": n_g}


if __name__ == "__main__":
    print("=== Buhlmann credibility (1967) ===\n")
    rng = np.random.default_rng(0)
    G = 60
    n_per = rng.integers(3, 15, size=G)                # small groups -> Z varies
    theta_g = rng.normal(loc=100, scale=6, size=G)    # small between-group SD
    obs_x = []; obs_g = []
    for g in range(G):
        obs_x.extend(rng.normal(loc=theta_g[g], scale=15, size=n_per[g]))
        obs_g.extend([g] * n_per[g])
    obs_x = np.array(obs_x); obs_g = np.array(obs_g)

    r = buhlmann_estimator(obs_g, obs_x)
    print(f"  Grand mean         = {r['grand_mean']:.2f}")
    print(f"  Within-group var   = {r['sigma2']:.2f}   (truth 225)")
    print(f"  Between-group var  = {r['tau2']:.2f}     (truth 36)")
    print(f"  K = sigma^2/tau^2   = {r['K']:.3f}")

    #  MSE against truth
    mse_raw = float(np.mean((r['raw_group_mean'] - theta_g) ** 2))
    mse_cred = float(np.mean((r['prediction_by_group'] - theta_g) ** 2))
    print(f"\n  MSE(raw group mean vs true theta_g)   = {mse_raw:.2f}")
    print(f"  MSE(Buhlmann prediction vs theta_g)    = {mse_cred:.2f}   "
          f"(shrinkage cuts MSE by {(1 - mse_cred / mse_raw) * 100:.0f} %)")

    print(f"\n  Credibility ranges: Z min = {r['Z_by_group'].min():.2f}, "
          f"max = {r['Z_by_group'].max():.2f}   (higher n_g -> higher Z)")

    print("\n--- library cross-check (actuar / ChainLadder R; chainladder Python) ---")

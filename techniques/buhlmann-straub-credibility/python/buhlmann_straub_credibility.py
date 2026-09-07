"""Buhlmann-Straub credibility (Reference Sec 24.21).

Buhlmann & Straub 1970 'Credibility for loss ratios'. Extension of
Buhlmann's model that allows GROUP-SPECIFIC EXPOSURES (or weights)
w_g, giving different variance per group:

    Var(X | g) = sigma^2 / w_g       (heteroskedastic within groups)

Credibility factor per group:

    Z_g = w_g / (w_g + K)        K = sigma^2 / tau^2
    P_g = Z_g * X_bar_g + (1 - Z_g) * mu

where X_bar_g is the WEIGHTED mean using w_g and mu is the collective
weighted mean.

Common actuarial use: policy-year exposures where large policies get
more credibility than small ones with the same number of claims.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def buhlmann_straub(group_ids, X, w):
    """Fit Buhlmann-Straub model with per-obs weights.

    group_ids : integer group per observation
    X          : observation values (loss ratios etc.)
    w          : observation weights (exposure)
    """
    G = int(group_ids.max()) + 1
    w_g = np.array([w[group_ids == g].sum() for g in range(G)])
    mean_g = np.array([np.sum(w[group_ids == g] * X[group_ids == g]) / w_g[g] for g in range(G)])
    grand_mean = float((w * X).sum() / w.sum())

    #  Within-group variance sigma^2
    ss_within = 0.0; df_within = 0
    for g in range(G):
        idx = group_ids == g
        if idx.sum() > 1:
            resid = X[idx] - mean_g[g]
            ss_within += float(np.sum(w[idx] * resid ** 2))
            df_within += idx.sum() - 1
    sigma2 = ss_within / max(df_within, 1)

    #  Between-group variance tau^2
    W = w_g.sum()
    var_group = float(np.sum(w_g * (mean_g - grand_mean) ** 2) / (G - 1))
    tau2 = max(var_group - sigma2 / (W / G), 0.0)
    K = sigma2 / tau2 if tau2 > 0 else np.inf
    Z = w_g / (w_g + K)
    pred = Z * mean_g + (1 - Z) * grand_mean
    return {"grand_mean": grand_mean, "sigma2": sigma2, "tau2": tau2, "K": float(K),
            "Z": Z, "prediction": pred, "raw_mean": mean_g, "w_g": w_g}


if __name__ == "__main__":
    print("=== Buhlmann-Straub credibility (per-group exposure) ===\n")
    rng = np.random.default_rng(0)
    G = 40
    #  Group true means around mu = 1
    theta_g = rng.normal(loc=1.0, scale=0.15, size=G)
    #  Exposures vary WIDELY (some tiny, some medium) -> Z varies
    exposure = rng.gamma(shape=0.6, scale=1.0, size=G) * 3
    obs_x = []; obs_g = []; obs_w = []
    for g in range(G):
        n_g = rng.integers(3, 8)
        w_i = max(exposure[g] / n_g, 0.05)
        obs_x.extend(rng.normal(loc=theta_g[g], scale=0.6 / np.sqrt(w_i), size=n_g))
        obs_g.extend([g] * n_g)
        obs_w.extend([w_i] * n_g)
    obs_x = np.array(obs_x); obs_g = np.array(obs_g); obs_w = np.array(obs_w)

    r = buhlmann_straub(obs_g, obs_x, obs_w)
    print(f"  n groups = {G}, weighted grand mean = {r['grand_mean']:.3f}")
    print(f"  sigma^2 = {r['sigma2']:.3f}, tau^2 = {r['tau2']:.3f}, K = {r['K']:.2f}")
    print(f"  Z ranges: {r['Z'].min():.2f} to {r['Z'].max():.2f}   (bigger exposures -> bigger Z)")

    mse_raw = float(np.mean((r["raw_mean"] - theta_g) ** 2))
    mse_bs = float(np.mean((r["prediction"] - theta_g) ** 2))
    print(f"\n  MSE(raw group mean vs truth)      = {mse_raw:.4f}")
    print(f"  MSE(Buhlmann-Straub prediction)   = {mse_bs:.4f}   "
          f"(shrinkage cuts MSE by {(1 - mse_bs / mse_raw) * 100:.0f} %)")

    print("\n--- library cross-check (actuar::cm R; chainladder Python) ---")

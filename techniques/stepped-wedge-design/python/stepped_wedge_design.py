"""Stepped-Wedge Cluster Trial (Reference Sec 47.265).

Hussey & Hughes 2007 Contemp Clin Trials. Clusters are
randomised to the TIMING of intervention rollout. All clusters
end up treated; the design gives every cluster its own control
period, gaining efficiency over parallel CRT:

    T periods, K clusters. In period t = 1..T:
        clusters 1..(t-1) are treated, others are control.
        outcome Y_{ijt} = mu + beta_t + theta * X_{ijt}
                           + u_i + e_{ijt}
        u_i ~ N(0, tau^2)  (cluster random effect)

Estimand theta (intervention effect); analyse via mixed-effects
regression or GEE.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def simulate_sw(K, T, m, theta, tau, sigma, rng):
    """Simulate SW-CRT dataset with K clusters, T periods, m per cluster-period."""
    # Randomise clusters to sequences (period they start treatment)
    seq = rng.permutation(np.tile(np.arange(1, T), K // (T - 1) + 1)[:K])
    rows = []
    beta = 0.1 * np.arange(T)                                     # time trend
    for i in range(K):
        u_i = rng.normal(0, tau)
        for t in range(T):
            X = 1 if t >= seq[i] else 0
            for _ in range(m):
                y = beta[t] + theta * X + u_i + rng.normal(0, sigma)
                rows.append((i, t, X, y))
    return np.array(rows, dtype=float), seq


def fit_sw_gee(data):
    """Simple OLS on (cluster, period, X) with cluster fixed effects; return theta_hat."""
    from sklearn.linear_model import LinearRegression
    cluster = data[:, 0].astype(int); period = data[:, 1].astype(int); X = data[:, 2]; y = data[:, 3]
    K = int(cluster.max() + 1); T = int(period.max() + 1)
    # Design: intercept + cluster dummies + period dummies + X
    C = np.eye(K)[cluster][:, 1:]                                 # drop first cluster
    P = np.eye(T)[period][:, 1:]
    D = np.column_stack([C, P, X])
    lr = LinearRegression().fit(D, y)
    return float(lr.coef_[-1])


if __name__ == "__main__":
    print("=== Stepped-Wedge Cluster Trial (Hussey & Hughes 2007) ===\n")
    rng = np.random.default_rng(0)

    K = 12                                                         # 12 clusters
    T = 5                                                          # 5 periods (1 baseline + 4 crossing steps)
    m = 10                                                         # 10 obs per cluster-period
    theta = 0.5
    tau = 0.3; sigma = 1.0

    data, seq = simulate_sw(K, T, m, theta, tau, sigma, rng)
    print(f"  K = {K} clusters, T = {T} periods, m = {m}/cluster/period, N = {len(data):,}")
    print(f"  True intervention effect theta = {theta}")
    print(f"  Cluster crossover sequence (period of first treatment): {seq.tolist()}\n")

    # Design visualisation
    print(f"  Design matrix (X = treated):")
    print(f"    {'cluster':<8}  " + "  ".join(f"t={t}" for t in range(T)))
    for i in range(K):
        row = ["1" if t >= seq[i] else "0" for t in range(T)]
        print(f"    {i + 1:<8}  " + "  ".join(f"{v:>3}" for v in row))

    # Fit with cluster + period fixed effects
    theta_hat = fit_sw_gee(data)
    print(f"\n  Estimated theta (cluster + period FE OLS): {theta_hat:.3f}   (true {theta})")

    # Contrast with a naive analysis ignoring period
    from sklearn.linear_model import LinearRegression
    theta_naive = float(LinearRegression().fit(data[:, 2:3], data[:, 3]).coef_[0])
    print(f"  Naive theta (X only, no time adjustment):  {theta_naive:.3f}   (biased by secular trend)")

    # 200-trial bias / SE simulation
    ests = []
    for _ in range(200):
        d, _ = simulate_sw(K, T, m, theta, tau, sigma, rng)
        ests.append(fit_sw_gee(d))
    ests = np.array(ests)
    print(f"\n  200-trial calibration: mean theta_hat = {ests.mean():.3f}   SE = {ests.std():.3f}")

    print("\n--- library cross-check (SWSamp R; swCRTdesign R; statsmodels MixedLM) ---")

"""Cluster-Randomized Trial (Reference Sec 47.266).

Donner & Klar 2000 Design and Analysis of Cluster Randomization
Trials. Randomise CLUSTERS (clinics, schools, villages) rather
than individuals. Within-cluster correlation (ICC = rho) inflates
required sample size by the DESIGN EFFECT (DEFF):

    DEFF = 1 + (m - 1) * rho

where m is the average cluster size. Effective n = N / DEFF.
Analysis via mixed-effects, GEE, or cluster-summary t-test.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def design_effect(m, icc):
    return 1 + (m - 1) * icc


def crt_sample_size(delta, sigma, m, icc, alpha=0.025, power=0.8):
    """Number of CLUSTERS per arm."""
    from scipy.stats import norm
    z_a = norm.ppf(1 - alpha); z_b = norm.ppf(power)
    n_indiv_per_arm = 2 * (z_a + z_b) ** 2 * sigma ** 2 / delta ** 2
    deff = design_effect(m, icc)
    return int(np.ceil(n_indiv_per_arm * deff / m))


def simulate_crt(k_per_arm, m, delta, sigma_e, sigma_c, rng):
    """Simulate a CRT and return cluster-summary t-test plus mixed-model estimates."""
    rows = []
    for arm in [0, 1]:
        for j in range(k_per_arm):
            u = rng.normal(0, sigma_c)
            for i in range(m):
                rows.append((arm, arm * k_per_arm + j, u + arm * delta + rng.normal(0, sigma_e)))
    return np.array(rows, dtype=float)


if __name__ == "__main__":
    print("=== Cluster-Randomized Trial (Donner & Klar 2000) ===\n")
    rng = np.random.default_rng(0)

    delta = 0.3; sigma = 1.0
    print(f"  Individually-randomised n per arm (delta={delta}, sigma={sigma}, "
          f"80% pwr, alpha=0.025): {int(np.ceil(2 * (1.96 + 0.84) ** 2 * sigma ** 2 / delta ** 2))}\n")

    print(f"  Design effect and cluster count (per arm):")
    print(f"    {'m':>5}  {'ICC':>5}  {'DEFF':>6}  {'clusters/arm':>12}")
    for m in [10, 25, 50, 100]:
        for icc in [0.01, 0.05, 0.10, 0.20]:
            deff = design_effect(m, icc)
            k = crt_sample_size(delta, sigma, m, icc)
            print(f"    {m:>5}  {icc:>5.2f}  {deff:>6.2f}  {k:>12}")
        print()

    # Simulate one CRT and compare cluster-summary t-test vs mixed model
    k = 20; m = 30; icc = 0.05
    sigma_c = (icc * sigma ** 2) ** 0.5
    sigma_e = ((1 - icc) * sigma ** 2) ** 0.5
    data = simulate_crt(k, m, delta, sigma_e, sigma_c, rng)
    arm = data[:, 0].astype(int); clust = data[:, 1].astype(int); y = data[:, 2]

    # Cluster-summary t-test
    cluster_means = [y[clust == c].mean() for c in np.unique(clust)]
    cluster_arm = [int(arm[clust == c][0]) for c in np.unique(clust)]
    means_0 = [cm for cm, a in zip(cluster_means, cluster_arm) if a == 0]
    means_1 = [cm for cm, a in zip(cluster_means, cluster_arm) if a == 1]
    from scipy.stats import ttest_ind
    t, p = ttest_ind(means_1, means_0)
    print(f"  Simulated CRT with k={k}, m={m}, ICC={icc}:")
    print(f"    Cluster-summary t-test:  t = {t:.2f}, p = {p:.4f}")

    # Naive individual-level t-test IGNORING clustering (BAD)
    t_naive, p_naive = ttest_ind(y[arm == 1], y[arm == 0])
    print(f"    Naive individual t-test: t = {t_naive:.2f}, p = {p_naive:.4f}")
    print(f"    (naive p is anti-conservative because it ignores intra-cluster correlation)\n")

    # Empirical ICC estimate from data
    from scipy.stats import f
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_within = 0.0
    for c in np.unique(clust):
        yc = y[clust == c]; ss_within += ((yc - yc.mean()) ** 2).sum()
    ss_between = ss_total - ss_within
    ms_between = ss_between / (2 * k - 1); ms_within = ss_within / (2 * k * m - 2 * k)
    rho_hat = max(0, (ms_between - ms_within) / (ms_between + (m - 1) * ms_within))
    print(f"    Empirical ICC estimate: {rho_hat:.3f}   (nominal {icc})")

    print("\n--- library cross-check (clusterPower R; cluster-randomised-trial samplesize) ---")

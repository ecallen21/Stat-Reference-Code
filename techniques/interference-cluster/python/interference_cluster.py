"""Interference + cluster randomization (Reference Sec 44.6).

SUTVA (stable unit treatment value assumption) fails when treatment
on one unit affects other units' outcomes.  Common in
marketplaces, social networks, ride-sharing.

Two design fixes:

  CLUSTER RANDOMIZATION -- randomise entire clusters (city,
    ride-share zone, social community).  Analysis: cluster-mean
    two-sample test; correct SE by design-effect.

  SWITCHBACK / TIME-BASED -- alternate treatment across time
    slots within the same market so each unit contributes both
    control and treatment periods.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def naive_ttest(y_C, y_T):
    return float(stats.ttest_ind(y_C, y_T, equal_var=False).pvalue)


def cluster_mean_ttest(clusters, y, assignment):
    """Compute per-cluster mean, then two-sample t-test on cluster means."""
    means = []; z = []
    for c in np.unique(clusters):
        mask = clusters == c
        means.append(y[mask].mean()); z.append(assignment[mask][0])
    means = np.array(means); z = np.array(z)
    return float(stats.ttest_ind(means[z == 1], means[z == 0], equal_var=False).pvalue)


if __name__ == "__main__":
    print("=== Interference + cluster randomization ===\n")
    rng = np.random.default_rng(0)
    # 20 clusters, 50 users each; effect = 0.20 at cluster level
    n_clusters, per_cluster = 20, 50
    n = n_clusters * per_cluster
    clusters = np.repeat(np.arange(n_clusters), per_cluster)
    # Cluster-randomised
    z_cluster = rng.integers(0, 2, n_clusters)
    assignment = z_cluster[clusters]
    cluster_effects = rng.normal(0, 0.6, n_clusters)      # unmeasured cluster-level noise
    y = 5 + 0.20 * assignment + cluster_effects[clusters] + rng.normal(0, 0.3, n)

    p_naive = naive_ttest(y[assignment == 0], y[assignment == 1])
    p_cluster = cluster_mean_ttest(clusters, y, assignment)
    print(f"  Naive per-user t-test  p = {p_naive:.3e}   (ignores clustering; over-precise)")
    print(f"  Cluster-mean t-test    p = {p_cluster:.3e}  (correct)")

    # Design effect ~ 1 + (m - 1) * ICC
    icc_est = 1 - (y - np.array([y[clusters == c].mean() for c in clusters])).var() / y.var()
    print(f"  Estimated ICC ~= {icc_est:.3f}")
    print(f"  Design effect ~= {1 + (per_cluster - 1) * icc_est:.2f}\n")

    print("  Switchback alternative: alternate treatment across time slots so each")
    print("  cluster contributes both control and treatment periods.\n")
    print("--- library cross-check (R inferference, DeclareDesign; Python inferference via rpy2) ---")

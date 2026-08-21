"""Two-stage cluster sampling (Reference §3.x extra).

Population: M primary sampling units (PSUs, e.g. schools) each containing
N_i secondary units (SSUs, e.g. students).

Stage 1: sample m PSUs (usually SRS or PPS).
Stage 2: within each sampled PSU i sample n_i SSUs.

Unbiased mean estimator (equal-probability SRS at both stages, with
approximately equal cluster sizes N):

    y_bar = (1 / m) * sum_i y_bar_i        (mean of cluster means)

Variance under two-stage SRS (Cochran 1977):

    Var(y_bar) = (1 - m/M) * S_b^2 / m
                + (1/m) * mean_i (1 - n_i/N_i) * S_w_i^2 / n_i

  * S_b^2: between-cluster variance of the true PSU means.
  * S_w_i^2: within-cluster variance in cluster i.

Design effect (DEFF):
    DEFF = Var_actual(y_bar) / Var_SRS(y_bar)   ~ 1 + (n_bar - 1) * rho
where rho is the intraclass correlation.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def two_stage_srs_mean(y_by_cluster, M: int = None, N_i: dict = None) -> dict:
    """y_by_cluster: dict {psu_id: array of SSU values} for the sampled clusters."""
    clusters = list(y_by_cluster.keys())
    m = len(clusters)
    means = np.array([y_by_cluster[c].mean() for c in clusters])
    within_var = np.array([y_by_cluster[c].var(ddof=1) for c in clusters])
    n_i = np.array([len(y_by_cluster[c]) for c in clusters])
    y_bar = float(means.mean())

    # between-cluster variance (of cluster means)
    S_b2 = float(np.var(means, ddof=1))
    # variance components
    fpc1 = 1 - m / M if M is not None else 1.0
    fpc2 = np.array([1 - n_i[k] / N_i[clusters[k]] if (N_i and clusters[k] in N_i) else 1.0
                     for k in range(m)])
    V_between = fpc1 * S_b2 / m
    V_within = (1.0 / m) * float((fpc2 * within_var / n_i).mean())
    Var = V_between + V_within

    # ICC estimator: ICC = (MS_B - MS_W) / (MS_B + (n_bar - 1) MS_W)
    n_bar = n_i.mean()
    grand = np.concatenate([y_by_cluster[c] - y_bar for c in clusters]).astype(float)
    total_ss = float((grand ** 2).sum())
    within_ss = sum(((y_by_cluster[c] - means[k]) ** 2).sum() for k, c in enumerate(clusters))
    between_ss = total_ss - within_ss
    MS_B = between_ss / (m - 1) if m > 1 else 0.0
    MS_W = within_ss / (sum(n_i) - m) if sum(n_i) > m else 0.0
    icc = (MS_B - MS_W) / (MS_B + (n_bar - 1) * MS_W) if MS_B + (n_bar - 1) * MS_W > 0 else 0.0

    # design effect approximation
    deff = 1 + (n_bar - 1) * icc

    return {"y_bar": y_bar, "SE": float(np.sqrt(Var)),
            "V_between": float(V_between), "V_within": float(V_within),
            "n_clusters": m, "n_ssu_total": int(sum(n_i)),
            "ICC": float(icc), "design_effect": float(deff),
            "method": "two-stage SRS mean + variance decomposition"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Population: M=200 clusters, each with N=50 units. Cluster means ~ N(70, 8^2),
    # within-cluster residual ~ N(0, 4^2)
    M = 200; N = 50
    true_cluster_means = rng.normal(70, 8, M)
    # sample m=15 clusters, then n_i=10 units from each
    m = 15; n_i = 10
    picked = rng.choice(M, m, replace=False)
    y_by_cluster = {int(c): true_cluster_means[c] + rng.normal(0, 4, n_i) for c in picked}
    N_i = {int(c): N for c in picked}

    res = two_stage_srs_mean(y_by_cluster, M=M, N_i=N_i)
    print(f"=== Two-stage SRS (M=200 PSUs, N=50 SSUs each; sample m=15, n_i=10) ===")
    print(f"  estimated mean y_bar = {res['y_bar']:.3f}   true = 70")
    print(f"  SE = {res['SE']:.3f}   V_between = {res['V_between']:.3f}, "
          f"V_within = {res['V_within']:.3f}")
    print(f"  intraclass correlation (ICC) = {res['ICC']:.3f}")
    print(f"  design effect (~ 1 + (n_bar - 1) * ICC) = {res['design_effect']:.2f}")

    # naive SRS SE for comparison
    all_y = np.concatenate(list(y_by_cluster.values()))
    srs_se = float(all_y.std(ddof=1) / np.sqrt(len(all_y)))
    print(f"\n  naive SRS SE (ignores clustering)     = {srs_se:.3f}")
    print(f"  effective sample size n_eff = n / DEFF = {int(len(all_y) / res['design_effect'])}")

    print("\n--- library cross-check (R survey::svymean with two-stage design) ---")

"""Coreset selection (Reference Sec 47.44).

Feldman & Langberg 2011 'A unified framework for approximating and
clustering data'; Mirzasoleiman et al 2020 'Coresets for data-
efficient training of machine learning models', ICML. A coreset is
a small weighted subset S of the training set such that any
model's loss on S approximates its loss on the full data:

    |L(f; S, w) - L(f; D)|  <=  epsilon * L(f; D)

For k-means, Feldman-Langberg sensitivity sampling picks x_i with
probability proportional to its 'sensitivity', an upper bound on
the worst-case share of cost the point can cause.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kmeans_cost(X, C, w=None):
    """Weighted k-means (squared) cost."""
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    cost = d2.min(axis=1)
    if w is None:
        return float(cost.sum())
    return float((w * cost).sum())


def sensitivity_coreset_kmeans(X, k, m, rng=None):
    """Feldman-Langberg-style sensitivity sampling for k-means coresets.

    1. Approximate solution via k-means++ on the full data.
    2. Sensitivity  s_i = 2 * alpha * d(x_i, C)^2 / mean(d^2)
                      + 4 * alpha * cluster_cost / (|B_i| * mean(d^2))
       (constant alpha absorbed).
    3. Sample m points with prob proportional to s_i and weight 1 / (m * p_i).
    """
    rng = rng or np.random.default_rng(0)
    n = len(X)
    # step 1: k-means++ seeds only, no Lloyd needed for the bound
    idx = [int(rng.integers(n))]
    for _ in range(k - 1):
        d2 = ((X[:, None, :] - X[np.array(idx)][None, :, :]) ** 2).sum(-1).min(axis=1)
        idx.append(int(rng.choice(n, p=d2 / d2.sum())))
    C = X[np.array(idx)]
    d2_pt = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    assign = d2_pt.argmin(axis=1)
    d2_pt_min = d2_pt.min(axis=1)
    cluster_cost = np.array([d2_pt_min[assign == j].sum() for j in range(k)])
    cluster_size = np.array([max((assign == j).sum(), 1) for j in range(k)])
    mean_cost = d2_pt_min.mean() + 1e-12
    s = (2 * d2_pt_min / mean_cost
         + 4 * cluster_cost[assign] / (cluster_size[assign] * mean_cost))
    p = s / s.sum()
    sel = rng.choice(n, size=m, replace=False, p=p)
    w = 1.0 / (m * p[sel])
    return {"idx": sel, "w": w}


if __name__ == "__main__":
    print("=== Coreset selection (Feldman-Langberg 2011 sensitivity sampling) ===\n")
    rng = np.random.default_rng(0)
    n, d, k = 5000, 2, 5
    centers = rng.uniform(-5, 5, size=(k, d))
    X = np.vstack([c + 0.5 * rng.normal(size=(n // k, d)) for c in centers])

    # Baseline full-data k-means via Lloyd
    def lloyd(X, k, iters=30, seed=1):
        r = np.random.default_rng(seed)
        C = X[r.choice(len(X), k, replace=False)]
        for _ in range(iters):
            a = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1).argmin(1)
            C = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        return C
    C_full = lloyd(X, k)
    cost_full = kmeans_cost(X, C_full)
    print(f"  Full-data k-means cost (n={len(X)}): {cost_full:.1f}")

    for m in [50, 200, 500]:
        cs = sensitivity_coreset_kmeans(X, k, m, rng)
        Xc = X[cs["idx"]]; wc = cs["w"]
        # Lloyd on weighted coreset
        C_cs = X[cs["idx"][:k]]
        for _ in range(30):
            a = ((Xc[:, None, :] - C_cs[None, :, :]) ** 2).sum(-1).argmin(1)
            C_cs = np.array([np.average(Xc[a == j], weights=wc[a == j], axis=0)
                             if (a == j).any() else C_cs[j] for j in range(k)])
        cost_cs = kmeans_cost(X, C_cs)
        rel = abs(cost_cs - cost_full) / cost_full
        print(f"  Coreset m={m:4d}: cost on full data = {cost_cs:7.1f}  rel err = {rel:.3%}")

    print("\n--- library cross-check (submodlib / kmc2 Python; coreset R) ---")

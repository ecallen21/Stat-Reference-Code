"""Random survival forest (RSF) (Reference Sec 11.24).

Ishwaran, Kogalur, Blackstone & Lauer 2008 'Random survival forests',
Annals of Applied Statistics. Extension of Breiman's random forests to
right-censored survival data.

Key ingredients:
  * Bootstrap sample at each tree (63.2% in-bag, 36.8% OOB).
  * At each split, random subset of mtry covariates considered.
  * Split criterion: LOG-RANK statistic (or its variants), maximised
    over candidate cut points.
  * At each terminal node: NELSON-AALEN cumulative hazard.
  * Ensemble prediction: average leaf CHFs across trees.

We implement a minimal, vectorised RSF suitable for compact demos.
For production use randomForestSRC (R) or scikit-survival (Python).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def logrank_fast(t_sorted, e_sorted, in_left_sorted):
    """Vectorised two-sample log-rank on already-sorted arrays."""
    n = len(t_sorted)
    at_risk_total = np.arange(n, 0, -1)
    at_risk_left = np.cumsum(in_left_sorted[::-1])[::-1]
    O_l = 0.0; E_l = 0.0; V_l = 0.0
    #  Group ties: iterate by unique event times
    unique_t = np.unique(t_sorted[e_sorted == 1])
    for ut in unique_t:
        mask = t_sorted == ut
        d = e_sorted[mask].sum()
        if d == 0:
            continue
        i0 = int(np.searchsorted(t_sorted, ut))       # first index with t >= ut
        n_here = at_risk_total[i0]
        n_l = at_risk_left[i0]
        n_r = n_here - n_l
        d_l = e_sorted[mask & (in_left_sorted == 1)].sum()
        e_l = d * n_l / n_here if n_here > 0 else 0
        v_l = d * n_l * n_r * (n_here - d) / (n_here ** 2 * (n_here - 1)) if n_here > 1 else 0
        O_l += d_l; E_l += e_l; V_l += v_l
    return (O_l - E_l) ** 2 / V_l if V_l > 0 else 0.0


def best_split(X, t, e, mtry, rng, n_cuts=8):
    """Try mtry random covariates and n_cuts quantile cuts each."""
    n, p = X.shape
    cols = rng.choice(p, size=min(mtry, p), replace=False)
    order = np.argsort(t)
    t_s = t[order]; e_s = e[order]
    best = (None, None, -1.0)
    for j in cols:
        xj = X[:, j]
        qs = np.quantile(xj, np.linspace(0.15, 0.85, n_cuts))
        for c in np.unique(qs):
            in_left = (xj <= c).astype(int)[order]
            if in_left.sum() < 6 or (1 - in_left).sum() < 6:
                continue
            s = logrank_fast(t_s, e_s, in_left)
            if s > best[2]:
                best = (j, c, s)
    return best


def build_tree(X, t, e, mtry, min_leaf, max_depth, rng, depth=0):
    n = len(t)
    node = {"t": t, "e": e}
    if n < 2 * min_leaf or depth >= max_depth or e.sum() < 1:
        return node
    j, c, s = best_split(X, t, e, mtry, rng)
    if j is None or s <= 0:
        return node
    left = X[:, j] <= c
    if left.sum() < min_leaf or (~left).sum() < min_leaf:
        return node
    node.update({"split_j": j, "split_c": c,
                 "left":  build_tree(X[left], t[left], e[left], mtry, min_leaf, max_depth, rng, depth + 1),
                 "right": build_tree(X[~left], t[~left], e[~left], mtry, min_leaf, max_depth, rng, depth + 1)})
    return node


def na_chf_at(t, e, grid):
    """Nelson-Aalen cumulative hazard at grid times."""
    if len(t) == 0:
        return np.zeros_like(grid)
    order = np.argsort(t); t = t[order]; e = e[order]
    at_risk = np.arange(len(t), 0, -1)
    inc = np.where(e == 1, 1.0 / at_risk, 0.0)
    cum = np.cumsum(inc)
    #  For each grid point, cumulative hazard = cum at last time <= g
    idx = np.searchsorted(t, grid, side="right") - 1
    out = np.where(idx >= 0, cum[np.clip(idx, 0, len(cum) - 1)], 0.0)
    return out


def predict_chf(tree, x, grid):
    node = tree
    while "split_j" in node:
        node = node["left"] if x[node["split_j"]] <= node["split_c"] else node["right"]
    return na_chf_at(node["t"], node["e"], grid)


def rsf_fit_predict(X, t, e, n_trees=40, mtry=None, min_leaf=8, max_depth=5, seed=0):
    n, p = X.shape
    mtry = mtry or max(1, int(np.sqrt(p)))
    rng = np.random.default_rng(seed)
    grid = np.linspace(np.min(t), np.max(t), 20)
    ens = np.zeros((n, len(grid)))
    for _ in range(n_trees):
        idx = rng.choice(n, size=n, replace=True)
        tree = build_tree(X[idx], t[idx], e[idx], mtry, min_leaf, max_depth, rng)
        for i in range(n):
            ens[i] += predict_chf(tree, X[i], grid)
    ens /= n_trees
    return grid, ens


def cindex(risk, t, e):
    """Vectorised Harrell C-index."""
    n = len(t)
    concord = 0; total = 0
    for i in range(n):
        if e[i] == 0:
            continue
        cmp_t = t > t[i]
        n_cmp = cmp_t.sum()
        if n_cmp == 0:
            continue
        total += n_cmp
        concord += (risk[i] > risk[cmp_t]).sum() + 0.5 * (risk[i] == risk[cmp_t]).sum()
    return concord / total if total > 0 else 0.5


if __name__ == "__main__":
    print("=== Random survival forest -- Ishwaran et al. ===\n")
    rng = np.random.default_rng(0)
    n = 250; p = 5
    X = rng.normal(size=(n, p))
    lp = 0.8 * X[:, 0] - 0.5 * X[:, 1]           # true effects on x0, x1
    scale = np.exp(-lp)
    t_true = rng.weibull(1.5, size=n) * scale
    c = rng.exponential(3.0, size=n)
    e = (t_true <= c).astype(int)
    t = np.minimum(t_true, c)

    grid, CHF = rsf_fit_predict(X, t, e, n_trees=30, mtry=3, min_leaf=12, seed=0)
    risk = CHF[:, -1]                            # cumulative hazard at largest grid time
    C = cindex(risk, t, e)
    print(f"  n={n}, censoring = {(1 - e.mean()) * 100:.1f}%")
    print(f"  Ensemble CHF grid ({len(grid)} points) computed for all {n} obs.")
    print(f"  Harrell C-index = {C:.3f}   (Cox baseline on x0/x1 ~ 0.65-0.72)")

    print("\n--- library cross-check (randomForestSRC R; scikit-survival Python) ---")

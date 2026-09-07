"""Causal forest -- Athey & Wager (Reference Sec 15.37).

Athey, Tibshirani & Wager 2019 'Generalized random forests', Ann Stat.
A random-forest generalisation that estimates the CONDITIONAL average
treatment effect (CATE) tau(x) = E[Y(1) - Y(0) | X = x] with valid
pointwise inference.

Key ideas:
  * Fit an 'honest' forest where the split-selection sample is
    disjoint from the estimation sample -- prevents overfitting bias.
  * Split criterion targets heterogeneity in treatment effect
    (variance of tau) rather than outcome variance.
  * At a query point x, weights alpha_i(x) come from how often i is a
    neighbour of x across trees.
  * tau_hat(x) solves a local moment equation:
      sum_i alpha_i(x) * (T_i - e_hat(x)) * (Y_i - m_hat(x) -
        (T_i - e_hat(x)) * tau_hat(x)) = 0
  * Doubly-robust (R-learner / partialling-out) form used.

We implement a minimal 'R-forest': partial out nuisance models
m(x) = E[Y|X] and e(x) = P(T|X), then estimate tau(x) by
kernel-weighted regression of residuals -- weights supplied by a
random forest whose splits target CATE heterogeneity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def r_residuals(Y, T, X, folds=5, seed=0):
    """Cross-fitted residuals Y - E[Y|X] and T - P(T|X) via a linear model.

    Real GRF uses a first-stage random forest; a linear fit is used
    here for a compact self-contained demo.
    """
    n = len(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    Y_res = np.zeros(n); T_res = np.zeros(n)
    fold_size = n // folds
    for f in range(folds):
        te = idx[f * fold_size: (f + 1) * fold_size if f < folds - 1 else n]
        tr = np.setdiff1d(idx, te)
        Xtr = np.c_[np.ones(len(tr)), X[tr]]
        Xte = np.c_[np.ones(len(te)), X[te]]
        by, _, _, _ = np.linalg.lstsq(Xtr, Y[tr], rcond=None)
        bt, _, _, _ = np.linalg.lstsq(Xtr, T[tr], rcond=None)
        Y_res[te] = Y[te] - Xte @ by
        T_res[te] = T[te] - Xte @ bt
    return Y_res, T_res


def grow_hetero_tree(X, Y_res, T_res, min_leaf, max_depth, rng, depth=0):
    """Grow one tree; split rule targets treatment-effect heterogeneity.

    Simple heuristic: at each split, pick the (feature, cut) that
    maximises the difference in tau_hat between children, where
    tau_hat in a node = sum(T_res * Y_res) / sum(T_res^2).
    """
    n = len(Y_res)
    if n < 2 * min_leaf or depth >= max_depth:
        return {"idx_leaf": True, "n": n}
    p = X.shape[1]
    #  Random subset of features
    cols = rng.choice(p, size=max(1, int(np.sqrt(p))), replace=False)
    best = (None, None, -1.0)
    for j in cols:
        qs = np.quantile(X[:, j], np.linspace(0.15, 0.85, 6))
        for c in np.unique(qs):
            left = X[:, j] <= c
            n_l = left.sum(); n_r = n - n_l
            if n_l < min_leaf or n_r < min_leaf:
                continue
            #  tau_hat in each child
            denom_l = np.sum(T_res[left] ** 2)
            denom_r = np.sum(T_res[~left] ** 2)
            if denom_l < 1e-8 or denom_r < 1e-8:
                continue
            tau_l = np.sum(T_res[left] * Y_res[left]) / denom_l
            tau_r = np.sum(T_res[~left] * Y_res[~left]) / denom_r
            #  Weighted-heterogeneity gain = n_l*n_r/n * (tau_l - tau_r)^2
            gain = n_l * n_r / n * (tau_l - tau_r) ** 2
            if gain > best[2]:
                best = (j, c, gain)
    if best[0] is None:
        return {"idx_leaf": True, "n": n}
    j, c, _ = best
    left = X[:, j] <= c
    return {"idx_leaf": False, "split_j": j, "split_c": c,
            "left":  grow_hetero_tree(X[left],  Y_res[left],  T_res[left],  min_leaf, max_depth, rng, depth + 1),
            "right": grow_hetero_tree(X[~left], Y_res[~left], T_res[~left], min_leaf, max_depth, rng, depth + 1)}


def tree_leaf_membership(tree, x):
    node = tree; idx = 0
    while not node["idx_leaf"]:
        idx = idx * 2 + (1 if x[node["split_j"]] > node["split_c"] else 0)
        node = node["right"] if x[node["split_j"]] > node["split_c"] else node["left"]
    return idx


def causal_forest_fit_predict(X, T, Y, X_new, n_trees=100, min_leaf=15, max_depth=6, seed=0):
    Y_res, T_res = r_residuals(Y, T, X)
    rng = np.random.default_rng(seed)
    n_new = len(X_new)
    tau_hat = np.zeros(n_new)
    for _ in range(n_new):
        pass
    #  For each query x0, aggregate weights alpha_i(x0) across trees
    for k, x0 in enumerate(X_new):
        weights = np.zeros(len(Y))
        for _ in range(n_trees):
            idx = rng.choice(len(Y), size=len(Y), replace=True)
            tree = grow_hetero_tree(X[idx], Y_res[idx], T_res[idx],
                                     min_leaf, max_depth, rng)
            leaf_x0 = tree_leaf_membership(tree, x0)
            #  which in-bag samples share the leaf?
            for i, ii in enumerate(idx):
                if tree_leaf_membership(tree, X[ii]) == leaf_x0:
                    weights[ii] += 1
        weights /= weights.sum() if weights.sum() > 0 else 1
        num = np.sum(weights * T_res * Y_res)
        den = np.sum(weights * T_res ** 2)
        tau_hat[k] = num / den if den > 1e-8 else 0.0
    return tau_hat


if __name__ == "__main__":
    print("=== Causal forest -- Athey-Wager GRF (compact demo) ===\n")
    rng = np.random.default_rng(0)
    n = 800; p = 3
    X = rng.normal(size=(n, p))
    #  Heterogeneous effect: tau(x) = 1 + 2 * x0
    tau = 1.0 + 2.0 * X[:, 0]
    e = 1 / (1 + np.exp(-(0.6 * X[:, 1])))          # mild confounding
    T = (rng.uniform(size=n) < e).astype(int)
    Y = 0.5 * X[:, 1] - 0.3 * X[:, 2] + tau * T + rng.normal(scale=0.5, size=n)

    #  Predict tau at a small grid to keep runtime bounded
    x0_grid = np.zeros((5, p))
    x0_grid[:, 0] = np.linspace(-1.5, 1.5, 5)
    tau_hat = causal_forest_fit_predict(X, T, Y, x0_grid,
                                        n_trees=30, min_leaf=25, max_depth=4)

    print(f"  {'x0':>6s}  {'tau_hat':>8s}  {'tau_true':>8s}")
    for x0v, tv in zip(x0_grid[:, 0], tau_hat):
        truth = 1.0 + 2.0 * x0v
        print(f"  {x0v:>+6.2f}  {tv:>+8.3f}  {truth:>+8.3f}")
    print("\n  Note: minimal implementation; production use grf (R) / econml (Python).")

    print("\n--- library cross-check (grf R; econml.grf.CausalForest Python) ---")

"""Clustered ICE curves for heterogeneous treatment effects (Sec 47.51).

Zhao & Hastie 2019/2020 'Causal interpretations of black-box models',
JBES. ICE (Individual Conditional Expectation) curves plot predicted
outcome as a function of a treatment feature for each unit
separately. Clustering the shapes reveals subgroups with different
counterfactual response profiles -- an interpretable summary of the
Conditional Average Treatment Effect (CATE).

Recipe:
  1. Fit any regression / classifier   yhat = f(x, t).
  2. For each subject i, build the ICE curve y_i(t) at a grid of t.
  3. K-means the curves (or difference-from-baseline profiles).
  4. Read subgroup mean curves + membership shares.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.ensemble import GradientBoostingRegressor    # black-box f
from sklearn.cluster import KMeans    # profile clusterer


def ice_curves(model, X, t_grid, t_col):
    """Return (n, len(t_grid)) matrix of ICE curves."""
    curves = np.zeros((len(X), len(t_grid)))
    Xc = X.copy()
    for j, t in enumerate(t_grid):
        Xc[:, t_col] = t
        curves[:, j] = model.predict(Xc)
    return curves


def cluster_ice(curves, k=3, seed=0, use_delta=True):
    """K-means on centered ICE profiles."""
    if use_delta:
        profiles = curves - curves.mean(axis=1, keepdims=True)
    else:
        profiles = curves
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(profiles)
    return {"labels": km.labels_, "centers": km.cluster_centers_,
            "inertia": float(km.inertia_)}


if __name__ == "__main__":
    print("=== Clustered ICE for CATE (Zhao-Hastie 2019) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(size=n)              # continuous covariate
    x2 = rng.binomial(1, 0.5, size=n)    # moderator (creates HTE)
    t = rng.uniform(-2, 2, size=n)       # continuous treatment intensity
    # True model: additive main + moderator * treatment quadratic
    y = 1.0 + 0.5 * x1 + 0.5 * t + (2.0 * x2 - 1.0) * t ** 2 + 0.3 * rng.normal(size=n)
    X = np.column_stack([x1, x2, t])

    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=0)
    model.fit(X, y)

    t_grid = np.linspace(-2, 2, 30)
    curves = ice_curves(model, X, t_grid, t_col=2)

    cl = cluster_ice(curves, k=2)
    labels = cl["labels"]
    for lbl in range(2):
        share = float((labels == lbl).mean())
        mean_shape = cl["centers"][lbl]
        curvature = float(mean_shape.max() - mean_shape.min())
        moderator_mean = float(x2[labels == lbl].mean())
        print(f"  cluster {lbl}: n={int((labels==lbl).sum()):3d} ({share:.2f})   "
              f"curvature = {curvature:+.3f}   mean(x2 in cluster) = {moderator_mean:.2f}")

    # Sanity: cluster assignment should recover x2
    from sklearn.metrics import adjusted_rand_score    # ARI of recovered vs truth
    ari = adjusted_rand_score(x2, labels)
    print(f"\n  ARI(cluster labels, true moderator x2) = {ari:.3f}")

    print("\n--- library cross-check (iml / ICEbox R; PyCEbox / dalex Python) ---")

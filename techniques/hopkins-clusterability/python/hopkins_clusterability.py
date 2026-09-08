"""Hopkins clusterability statistic (Reference Sec 47.73).

Hopkins & Skellam 1954 'A new method for determining the type of
distribution of plant individuals', Annals of Botany. Test whether
a dataset has meaningful CLUSTER STRUCTURE before running any
clustering:

    1. Sample m points {y_1, ..., y_m} uniformly from the bounding
       box of X.
    2. Sample m points {x_i1, ..., x_im} at random from X.
    3. u_i = dist(y_i, nearest_X)  and  w_i = dist(x_ij, nearest_other_X).
    4. H = sum u_i^d / (sum u_i^d + sum w_i^d)      in [0, 1].

Interpretation:
    H ~ 0.5  -- data are Poisson (no cluster structure).
    H > 0.75 -- highly clustered.
    H < 0.25 -- regularly spaced.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.spatial import cKDTree    # nearest-neighbour lookup


def hopkins(X, m=None, seed=0):
    """Hopkins statistic (m defaults to 10% of n or 100, whichever smaller)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    if m is None:
        m = min(max(int(0.1 * n), 5), 100)
    lo = X.min(axis=0); hi = X.max(axis=0)
    Y = rng.uniform(lo, hi, size=(m, d))
    tree = cKDTree(X)
    # Nearest X-point for each Y sample
    u, _ = tree.query(Y, k=1)
    # Nearest OTHER X-point for a random subset of X
    idx = rng.choice(n, size=m, replace=False)
    dists, ids = tree.query(X[idx], k=2)
    w = dists[:, 1]                       # the 1st is the point itself
    # Modern practical form (used by R clustertend / pyclustertend):
    # H = sum(u) / (sum(u) + sum(w)); original 1954 used u^d, but that
    # underflows for uniform data in higher dimensions.
    U = float(np.sum(u))
    W = float(np.sum(w))
    return {"H": float(U / (U + W)), "m": m, "d": d}


if __name__ == "__main__":
    print("=== Hopkins clusterability (Hopkins-Skellam 1954) ===\n")
    rng = np.random.default_rng(0)

    # 1) Uniform data -- no structure -> H ~ 0.5 (average over 20 seeds)
    Hs = []
    for s in range(20):
        rr = np.random.default_rng(s)
        X_unif = rr.uniform(size=(500, 3))
        Hs.append(hopkins(X_unif, seed=100 + s)["H"])
    print(f"  Uniform on [0,1]^3 (n=500, 20 seeds):  mean H = {np.mean(Hs):.3f}   "
          f"(target ~ 0.50)")

    # 2) Well-separated 3-cluster mixture -> H high
    Hs = []
    for s in range(20):
        rr = np.random.default_rng(s)
        centers = np.array([[0, 0], [5, 0], [2.5, 4]])
        Xc = np.vstack([c + 0.2 * rr.normal(size=(150, 2)) for c in centers])
        Hs.append(hopkins(Xc, seed=200 + s)["H"])
    print(f"  3-cluster mixture (n=450, 20 seeds):   mean H = {np.mean(Hs):.3f}   "
          f"(target > 0.75)")

    # 3) Regular grid -> H < 0.5
    Hs = []
    g = np.arange(0, 10, 0.5)
    Xg = np.array([[x, y] for x in g for y in g])
    for s in range(20):
        Hs.append(hopkins(Xg, seed=300 + s)["H"])
    print(f"  Regular 2D grid (n={len(Xg)}, 20 seeds):    mean H = {np.mean(Hs):.3f}   "
          f"(target < 0.25)")

    print("\n  Rule of thumb: H > 0.75 suggests cluster structure worth analysing.")
    print("  H ~ 0.5 => data are 'clusterless' -- k-means output will be arbitrary.")
    print("\n--- library cross-check (clustertend R; hopkins / pyclustertend Python) ---")

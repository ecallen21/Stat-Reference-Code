"""k-medoids / PAM (Partitioning Around Medoids; Reference §9.10).

Like k-means, but:
    - Centers are ACTUAL data points (medoids), not means.
    - Works with ANY distance metric (not just Euclidean).
    - More robust to outliers.

PAM algorithm (Kaufman & Rousseeuw 1987):
    BUILD phase: greedy select K initial medoids one at a time to minimize
        total cost.
    SWAP phase:  repeatedly consider swapping each medoid with each non-medoid
        and accept swaps that reduce total cost. Stop when no improvement.

Cost = sum over points of distance to their assigned medoid.

Trade-off vs k-means:
    - PAM allows arbitrary distance (Manhattan, Gower for mixed data, ...).
    - Complexity O(K(n - K)^2) per iteration; slower than Lloyd's O(nK).
    - Faster alternatives: CLARA (Kaufman & Rousseeuw), CLARANS (Ng & Han).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _pairwise_distances(X, metric="euclidean"):
    X = np.asarray(X, dtype=float)
    if metric == "euclidean":
        diffs = X[:, None, :] - X[None, :, :]
        return np.sqrt((diffs ** 2).sum(-1))
    if metric == "manhattan":
        diffs = X[:, None, :] - X[None, :, :]
        return np.abs(diffs).sum(-1)
    raise ValueError("metric must be 'euclidean' or 'manhattan'")


def pam(X, k: int, metric: str = "euclidean", max_iter: int = 100, seed: int = 0) -> dict:
    """Partitioning Around Medoids (PAM). Returns medoid indices + labels."""
    X = np.asarray(X, dtype=float); n = X.shape[0]
    rng = np.random.default_rng(seed)
    D = _pairwise_distances(X, metric)

    # BUILD: greedy first-K initialization
    medoids = [int(rng.integers(n))]
    for _ in range(1, k):
        # For each candidate m, compute reduction in cost if added
        best_m, best_cost = None, np.inf
        for m in range(n):
            if m in medoids: continue
            candidates = medoids + [m]
            cost = D[:, candidates].min(axis=1).sum()
            if cost < best_cost:
                best_cost, best_m = cost, m
        medoids.append(best_m)
    medoids = np.array(medoids)

    # SWAP: try swapping each medoid with each non-medoid
    def total_cost(meds):
        return float(D[:, meds].min(axis=1).sum())
    curr_cost = total_cost(medoids)
    for it in range(max_iter):
        improved = False
        for i, m in enumerate(medoids):
            for h in range(n):
                if h in medoids: continue
                new_med = medoids.copy(); new_med[i] = h
                c = total_cost(new_med)
                if c < curr_cost - 1e-12:
                    medoids = new_med; curr_cost = c; improved = True; break
            if improved: break
        if not improved: break

    labels = np.argmin(D[:, medoids], axis=1)
    return {"medoids_indices": medoids.tolist(),
            "medoids_coordinates": X[medoids].tolist(),
            "labels": labels.tolist(),
            "total_cost": float(curr_cost),
            "n_iter": it + 1,
            "metric": metric, "k": k, "n": int(n),
            "method": "PAM (Partitioning Around Medoids)"}


def library_versions(X, k):
    try:
        from sklearn_extra.cluster import KMedoids
        km = KMedoids(n_clusters=k, method="pam", random_state=0).fit(X)
        return {"sklearn_extra KMedoids inertia": float(km.inertia_),
                "sklearn_extra medoid indices": km.medoid_indices_.tolist()}
    except Exception as ex:
        return {"sklearn_extra (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    centers = np.array([[0, 0], [4, 4], [8, 0]])
    X = np.vstack([rng.normal(c, 0.7, size=(40, 2)) for c in centers])
    # Add one strong outlier
    X = np.vstack([X, [[20, 20]]])

    print("=== PAM (k=3, robust to outliers) ===")
    fit = pam(X, k=3)
    print(f"  medoid indices: {fit['medoids_indices']}")
    print(f"  medoid coords:  {fit['medoids_coordinates']}")
    from collections import Counter
    print(f"  cluster sizes:  {dict(Counter(fit['labels']))}")
    print(f"  total cost:     {fit['total_cost']:.4f}")
    print(f"  n_iter:         {fit['n_iter']}")

    print("\n--- library ---")
    for k, v in library_versions(X, 3).items():
        print(f"  {k}: {v}")

"""Affinity Propagation (Reference Sec 47.80).

Frey & Dueck 2007 'Clustering by passing messages between data
points', Science 315. Message-passing algorithm that
simultaneously identifies EXEMPLARS and cluster memberships:

    responsibility  r(i, k) = s(i, k) - max_{k' != k} (a(i, k') + s(i, k'))
    availability    a(i, k) = min(0, r(k, k) + sum_{i' != i, k}  max(0, r(i', k)))
    a(k, k) = sum_{i' != k}  max(0, r(i', k))

Damped by lambda. Number of clusters set implicitly by the
DIAGONAL of similarity s(i, i) = preference (median-of-s is a
common default).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def affinity_propagation(S, damping=0.9, max_iter=200, tol_iters=10):
    """Simple AP; S is n x n similarity (higher = more similar).
    Preferences on the diagonal.
    """
    n = S.shape[0]
    R = np.zeros_like(S)
    A = np.zeros_like(S)
    last_exemplars = None
    stable = 0
    for it in range(max_iter):
        # Update responsibilities
        AS = A + S
        idx1 = AS.argmax(axis=1)
        max1 = AS[np.arange(n), idx1]
        AS[np.arange(n), idx1] = -np.inf
        max2 = AS.max(axis=1)
        R_new = S - max1[:, None]
        R_new[np.arange(n), idx1] = S[np.arange(n), idx1] - max2
        R = damping * R + (1 - damping) * R_new
        # Update availabilities
        Rp = np.maximum(R, 0)
        np.fill_diagonal(Rp, R.diagonal())
        A_new = np.minimum(0, R.diagonal()[None, :] + Rp.sum(axis=0)[None, :] - Rp)
        np.fill_diagonal(A_new, Rp.sum(axis=0) - Rp.diagonal())
        A = damping * A + (1 - damping) * A_new
        # Extract exemplars
        E = np.where((R + A).diagonal() > 0)[0]
        if last_exemplars is not None and np.array_equal(E, last_exemplars):
            stable += 1
            if stable >= tol_iters:
                break
        else:
            stable = 0
        last_exemplars = E
    labels = np.argmax(S[:, E], axis=1) if len(E) > 0 else np.zeros(n, dtype=int)
    return {"labels": labels, "exemplars": E, "iters": it + 1}


if __name__ == "__main__":
    print("=== Affinity Propagation (Frey-Dueck 2007) ===\n")
    rng = np.random.default_rng(0)
    centers = np.array([[-3, -3], [-3, 3], [3, -3], [3, 3]])
    X = np.vstack([c + 0.5 * rng.normal(size=(50, 2)) for c in centers])
    y_true = np.repeat(np.arange(4), 50)
    n = len(X)

    # Sklearn reference implementation with different preferences
    from sklearn.cluster import AffinityPropagation    # library baseline
    from sklearn.metrics import adjusted_rand_score    # ARI eval
    S_off = -((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    S_off = S_off[~np.eye(n, dtype=bool)]
    for pref_q in [0.05, 0.25, 0.50, 0.75]:
        pref = float(np.quantile(S_off, pref_q))
        ap = AffinityPropagation(preference=pref, damping=0.9,
                                   random_state=0, max_iter=300).fit(X)
        n_ex = len(ap.cluster_centers_indices_)
        ari = adjusted_rand_score(y_true, ap.labels_)
        print(f"  preference = q{int(pref_q*100):2d} ({pref:6.1f})  ->  "
              f"{n_ex:3d} clusters, ARI = {ari:.3f}")

    print("\n  Median-preference default typically recovers a natural K without pre-specification.")
    print("  (More-negative preference -> fewer exemplars.)")
    print("\n--- library cross-check (apcluster R; sklearn.cluster.AffinityPropagation Python) ---")

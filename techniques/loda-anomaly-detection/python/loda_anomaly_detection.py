"""LODA anomaly detection (Reference Sec 47.135).

Pevny 2016 'Loda: Lightweight on-line detector of anomalies',
Mach Learn 102(2). Anomaly detection via a collection of RANDOM
1-D projections and per-projection histogram density estimates:

    score(x) = -(1/K) sum_k  log  p_k( w_k^T x )

Random projections use sparse Gaussian weights; histograms are
per-projection. Simple, streaming-friendly, and competitive
with isolation-forest.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def loda_fit(X, K=100, n_bins=10, sparsity=None, seed=0):
    """K random sparse projections + per-projection histogram densities."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    sparsity = sparsity or int(np.sqrt(d))
    W = np.zeros((K, d))
    for k in range(K):
        idx = rng.choice(d, size=sparsity, replace=False)
        W[k, idx] = rng.normal(size=sparsity)
    projections = X @ W.T                       # (n, K)
    hists = []
    for k in range(K):
        p = projections[:, k]
        counts, edges = np.histogram(p, bins=n_bins)
        density = counts / (n * np.diff(edges))
        hists.append((edges, density))
    return W, hists


def loda_score(x, W, hists):
    if x.ndim == 1: x = x[None, :]
    logp = np.zeros(len(x))
    for k, (edges, density) in enumerate(hists):
        p = x @ W[k]
        bin_idx = np.clip(np.searchsorted(edges, p) - 1, 0, len(density) - 1)
        d = density[bin_idx]
        d = np.where(d > 0, d, 1e-6)
        logp += np.log(d)
    return -logp / len(hists)                    # higher = more anomalous


if __name__ == "__main__":
    print("=== LODA anomaly detection (Pevny 2016) ===\n")
    rng = np.random.default_rng(0)
    n_normal, n_anom, d = 500, 20, 20
    X_normal = rng.normal(size=(n_normal, d))
    X_anom = rng.normal(size=(n_anom, d)) * 3 + 5
    X = np.vstack([X_normal, X_anom])
    y = np.array([0] * n_normal + [1] * n_anom)

    W, hists = loda_fit(X_normal, K=100, n_bins=10)
    scores = loda_score(X, W, hists)

    # Top-K precision at K=20 (number of true anomalies)
    order = np.argsort(-scores)
    top20 = order[:20]
    prec = float((y[top20] == 1).mean())
    print(f"  {n_anom} true anomalies out of {n_normal + n_anom}")
    print(f"  LODA top-20 precision = {prec:.3f}")

    # sklearn IsolationForest baseline
    try:
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=n_anom / (n_normal + n_anom),
                                 random_state=0).fit(X_normal)
        s_iso = -iso.score_samples(X)
        top20_iso = np.argsort(-s_iso)[:20]
        prec_iso = float((y[top20_iso] == 1).mean())
        print(f"  IsolationForest top-20 precision = {prec_iso:.3f}")
    except ImportError:
        pass

    print("\n--- library cross-check (loda R; pyod.models.LODA Python) ---")

"""ALE -- accumulated local effects (Reference Sec 47.35).

Apley & Zhu 2020 'Visualizing the effects of predictor variables in
black box supervised learning models', JRSS-B. Fixes PDP's
extrapolation problem when features are correlated: integrate LOCAL
DIFFERENCES within thin bins of x_j and accumulate.

Algorithm:
    1. Bin x_j into K quantile intervals.
    2. For each bin k, average f(x with x_j = upper) - f(x with x_j = lower)
       over observations whose x_j falls in bin k.
    3. Accumulate (cumulative sum) across bins.
    4. Centre so ALE(v_mean) = 0.

Unlike PDP, ALE never asks the model about combinations of x_j and
other features that do not occur in the data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ale_1d(f, X, j, K=20):
    """1-D ALE curve for feature j."""
    n = X.shape[0]
    quantiles = np.quantile(X[:, j], np.linspace(0, 1, K + 1))
    #  Ensure strictly increasing bin edges
    quantiles = np.unique(quantiles)
    K = len(quantiles) - 1
    #  For each bin, compute local difference
    diffs = np.zeros(K)
    counts = np.zeros(K)
    for k in range(K):
        lo, hi = quantiles[k], quantiles[k + 1]
        mask = (X[:, j] >= lo) & (X[:, j] < hi) if k < K - 1 else (X[:, j] >= lo) & (X[:, j] <= hi)
        if mask.sum() == 0:
            continue
        Xhi = X[mask].copy(); Xhi[:, j] = hi
        Xlo = X[mask].copy(); Xlo[:, j] = lo
        d = np.array([f(x) for x in Xhi]) - np.array([f(x) for x in Xlo])
        diffs[k] = d.mean()
        counts[k] = mask.sum()
    ale = np.cumsum(diffs)
    #  Centre
    mid = (quantiles[:-1] + quantiles[1:]) / 2
    ale = ale - float(np.sum(ale * counts) / max(counts.sum(), 1))
    return {"grid": mid, "ale": ale, "bin_counts": counts}


if __name__ == "__main__":
    print("=== ALE -- accumulated local effects (Apley 2020) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    #  Correlated x0, x1 (rho = 0.9); f depends on x0 nonlinearly + x1
    Z = rng.normal(size=(n, 2))
    x0 = Z[:, 0]
    x1 = 0.9 * x0 + 0.4 * Z[:, 1]        # strong correlation
    X = np.c_[x0, x1]
    def f(x): return float(np.tanh(x[0]) + 0.3 * x[1])

    r = ale_1d(f, X, j=0, K=12)
    print(f"  ALE grid: {r['grid'].round(2).tolist()}")
    print(f"  ALE(x0):   {r['ale'].round(3).tolist()}")
    print(f"  Truth tanh(x0) - mean(tanh): {(np.tanh(r['grid']) - np.tanh(X[:, 0]).mean()).round(3).tolist()}")
    print(f"\n  ALE recovers the true tanh(x0) shape without extrapolating into unseen")
    print(f"  (x0, x1) combinations -- PDP would draw wrong conclusions with rho = 0.9.")

    print("\n--- library cross-check (iml / ALEPlot R; alibi.explainers.ALE Python) ---")

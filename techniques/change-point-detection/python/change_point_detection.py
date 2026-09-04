"""Change-point detection (Reference Sec 38.8).

Retrospective (offline) detection of abrupt shifts in the mean of a
time series.  Two classical algorithms are implemented here:

  BINARY SEGMENTATION (Vostrikova 1981)
    Find the single best split under a CUSUM-style test, recurse on
    the two halves until no split beats a threshold.  O(n log n) but
    can miss adjacent change points.

  PELT (Killick-Fearnhead-Eckley 2012)
    Exact optimal partition under a cost + penalty, with a pruning
    step that removes provably-suboptimal candidates.  O(n) expected.

Cost function used: negative log-likelihood of a Gaussian mean shift
(minimum L2 residual).  Penalty (BIC-style): sigma_hat^2 * log(n).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _seg_cost(x, i, j):
    """L2 cost of segment x[i:j] under mean-shift Gaussian."""
    seg = x[i:j]
    if len(seg) == 0:
        return 0.0
    return float(((seg - seg.mean()) ** 2).sum())


def binary_segmentation(x, penalty, min_size=5):
    """Greedy recursive segmentation with a stopping penalty."""
    cps = []
    def _split(lo, hi):
        if hi - lo < 2 * min_size:
            return
        base = _seg_cost(x, lo, hi)
        best_gain = 0.0
        best = None
        for k in range(lo + min_size, hi - min_size + 1):
            g = base - (_seg_cost(x, lo, k) + _seg_cost(x, k, hi))
            if g > best_gain:
                best_gain, best = g, k
        if best is not None and best_gain > penalty:
            cps.append(best)
            _split(lo, best)
            _split(best, hi)
    _split(0, len(x))
    return sorted(cps)


def pelt(x, penalty, min_size=5):
    """Optimal partition via PELT (Killick-Fearnhead-Eckley 2012)."""
    n = len(x)
    F = np.full(n + 1, np.inf)
    F[0] = -penalty
    cp_at = [[] for _ in range(n + 1)]
    R = [0]                               # candidate change points
    for t in range(min_size, n + 1):
        best = np.inf
        best_prev = None
        for s in R:
            if t - s < min_size:
                continue
            v = F[s] + _seg_cost(x, s, t) + penalty
            if v < best:
                best = v; best_prev = s
        F[t] = best
        cp_at[t] = cp_at[best_prev] + ([best_prev] if best_prev > 0 else [])
        # PELT pruning
        R = [s for s in R if F[s] + _seg_cost(x, s, t) <= F[t]] + [t]
    return sorted(cp_at[n])


if __name__ == "__main__":
    print("=== Change-point detection: Binary Segmentation vs PELT ===\n")
    rng = np.random.default_rng(0)
    # Three true change points: 60, 150, 220
    means = [0.0, 2.0, -1.0, 3.0]
    sizes = [60, 90, 70, 80]
    x = np.concatenate([rng.normal(m, 1.0, n) for m, n in zip(means, sizes)])
    true_cp = np.cumsum(sizes)[:-1].tolist()

    sig2 = 1.0
    pen = 2.5 * sig2 * np.log(len(x))     # modified BIC penalty

    bs_cp = binary_segmentation(x, penalty=pen)
    pelt_cp = pelt(x, penalty=pen)

    print(f"  Series length n = {len(x)}, true change points at {true_cp}")
    print(f"  Binary segmentation: {bs_cp}")
    print(f"  PELT               : {pelt_cp}")

    print("\n--- library cross-check (R changepoint::cpt.mean; Python ruptures Pelt/Binseg) ---")

"""Dynamic Time Warping (Reference §13.22).

Similarity between two time series that may be misaligned in time.
Instead of Euclidean distance (which is destroyed by even one-sample
shifts), DTW finds the optimal monotone alignment path minimizing the
sum of pointwise distances.

Cost matrix D and cumulative cost C:
    D[i, j] = |x_i - y_j|
    C[0, 0] = D[0, 0]
    C[i, j] = D[i, j] + min(C[i-1, j], C[i, j-1], C[i-1, j-1])

DTW distance = C[N, M].  Optimal alignment path recovered by backtracking.

Sakoe-Chiba band constraint
    Restrict |i - j| <= w for a window w to prevent pathological warping
    and speed the algorithm from O(N M) to O(N w).

Applications
    - Speech recognition (originally).
    - Gesture / gait / activity classification.
    - Any classification of time series with local time distortions.
    - Distance for kNN-DTW classifier (see 'ts-features-classification').
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def dtw(x, y, window: int = None) -> dict:
    """Dynamic time warping distance + alignment path.

    x, y   : 1-D sequences (can have different lengths).
    window : Sakoe-Chiba band width in samples; None = no constraint.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    N, M = len(x), len(y)
    inf = float("inf")
    C = np.full((N + 1, M + 1), inf); C[0, 0] = 0.0
    if window is None: window = max(N, M)
    for i in range(1, N + 1):
        j_lo = max(1, i - window); j_hi = min(M, i + window)
        for j in range(j_lo, j_hi + 1):
            d = abs(x[i - 1] - y[j - 1])
            C[i, j] = d + min(C[i - 1, j], C[i, j - 1], C[i - 1, j - 1])
    # Backtrack
    path = [(N - 1, M - 1)]
    i, j = N, M
    while i > 1 or j > 1:
        candidates = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        vals = [C[a, b] for a, b in candidates]
        i, j = candidates[int(np.argmin(vals))]
        path.append((i - 1, j - 1))
    path.reverse()
    return {"distance": float(C[N, M]),
            "distance_normalized": float(C[N, M] / len(path)),
            "path": path,
            "cost_matrix_shape": (N, M),
            "window": int(window),
            "method": "Dynamic Time Warping (Sakoe-Chiba constrained)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Two shifted sinusoids (DTW should be much smaller than Euclidean) ===")
    N = 100
    t = np.linspace(0, 4 * math.pi, N)
    x = np.sin(t)
    y = np.sin(t + 0.6)
    euc = float(np.sqrt(np.sum((x - y) ** 2)))
    dtw_res = dtw(x, y)
    print(f"  Euclidean distance:            {euc:.3f}")
    print(f"  DTW distance:                  {dtw_res['distance']:.3f}")
    print(f"  DTW normalized (per-step):     {dtw_res['distance_normalized']:.3f}")

    print("\n=== Two random walks (unrelated) ===")
    x = np.cumsum(rng.normal(size=100))
    y = np.cumsum(rng.normal(size=100))
    print(f"  DTW distance: {dtw(x, y)['distance']:.3f}")

    print("\n=== Sakoe-Chiba constraint speedup (window = 5) ===")
    import time
    N = 500
    x = np.sin(np.linspace(0, 4 * math.pi, N)); y = np.sin(np.linspace(0.6, 4 * math.pi + 0.6, N))
    t0 = time.time(); d1 = dtw(x, y)["distance"]; t1 = time.time()
    t2 = time.time(); d2 = dtw(x, y, window=5)["distance"]; t3 = time.time()
    print(f"  full DTW      : distance = {d1:.3f}  ({t1 - t0:.3f} s)")
    print(f"  band w=5      : distance = {d2:.3f}  ({t3 - t2:.3f} s)")

    print("\n--- library cross-check (dtaidistance, if available) ---")
    try:
        from dtaidistance import dtw as dtaidtw
        x = np.sin(np.linspace(0, 4 * math.pi, 100)); y = np.sin(np.linspace(0.6, 4 * math.pi + 0.6, 100))
        print(f"  dtaidistance DTW: {dtaidtw.distance(x, y):.3f}")
    except Exception as ex:
        print(f"  (dtaidistance not available: {ex})")

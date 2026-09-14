"""Detrended Fluctuation Analysis / Hurst (Ref Sec 47.330).

Peng et al 1994 Phys Rev E. Estimate long-range correlations
in nonstationary time series:

    1. cumulative sum: Y(k) = sum_{i<=k}(x_i - mean)
    2. partition into boxes of size n; detrend within each box
    3. F(n) = sqrt(mean(residuals^2))
    4. F(n) ~ n^alpha; alpha is the DFA exponent

    alpha = 0.5   uncorrelated (white noise)
    alpha < 0.5   anti-correlated
    alpha > 0.5   long-range correlated (persistent)
    alpha = 1.0   1/f noise
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def dfa(x, scales=None):
    x = np.asarray(x, float) - np.mean(x)
    y = np.cumsum(x)
    N = len(y)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(8), np.log10(N // 4), 15).astype(int))
    fluct = []
    for n in scales:
        n_boxes = N // n
        F = 0.0
        for k in range(n_boxes):
            seg = y[k * n:(k + 1) * n]
            t = np.arange(n)
            p = np.polyfit(t, seg, 1)
            resid = seg - (p[0] * t + p[1])
            F += np.sum(resid ** 2)
        fluct.append(np.sqrt(F / (n_boxes * n)))
    fluct = np.array(fluct)
    # Linear fit in log-log
    log_n = np.log(scales); log_F = np.log(fluct)
    alpha, _ = np.polyfit(log_n, log_F, 1)
    return float(alpha), scales, fluct


if __name__ == "__main__":
    print("=== DFA / Hurst Exponent (Peng et al 1994) ===\n")
    rng = np.random.default_rng(0)

    N = 4000
    # White noise: alpha ~ 0.5
    x_white = rng.standard_normal(N)
    # Persistent (integrated white -> Brownian motion): alpha ~ 1.5
    x_brown = np.cumsum(rng.standard_normal(N))
    # 1/f noise (Voss algorithm approx via cumulative sum of subsegments)
    from scipy.signal import lfilter
    x_pink = lfilter([1], [1, -0.99], rng.standard_normal(N))
    # Anti-correlated: differenced noise, alpha ~ 0
    x_diff = np.diff(rng.standard_normal(N + 1))

    for name, x in [("white noise", x_white), ("1/f pink", x_pink),
                    ("Brownian motion", x_brown), ("differenced noise", x_diff)]:
        alpha, _, _ = dfa(x)
        print(f"  {name:>18}   DFA exponent alpha = {alpha:.3f}")

    print(f"\n  Reference values:")
    print(f"    white noise      alpha ~ 0.5")
    print(f"    1/f pink noise   alpha ~ 1.0")
    print(f"    Brownian motion  alpha ~ 1.5")
    print(f"    diff-noise       alpha ~ 0.0 (anti-persistent)")

    print("\n--- library cross-check (nolds.dfa; MFDFA; antropy.dfa) ---")

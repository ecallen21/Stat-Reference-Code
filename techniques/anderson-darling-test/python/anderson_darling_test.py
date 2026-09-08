"""Anderson-Darling GoF test (Reference Sec 47.101).

Anderson & Darling 1954 'A test of goodness of fit', JASA 49.
EDF-based test that puts more weight on the TAILS of the
distribution than Kolmogorov-Smirnov or Cramér-von Mises:

    A^2 = -n - (1/n) sum_{i=1}^n [ (2i-1) ( log U_(i) + log(1 - U_(n+1-i)) ) ]

with U_(i) = F_0(x_(i)) the ordered probability-integral-
transformed sample. Powerful against tail departures. Critical
values depend on which parameters (if any) were estimated from the
data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats    # asymptotic + tabulated critical values


def anderson_darling(x, dist="norm", args=(), estimate=True):
    """A^2 GoF against `dist` (name from scipy.stats).
    If estimate=True, ML-fit dist params from x (uses A^2* adjusted).
    """
    x = np.sort(np.asarray(x))
    n = len(x)
    d = getattr(stats, dist)
    if estimate:
        args = d.fit(x)
    U = d.cdf(x, *args)
    U = np.clip(U, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    A2 = -n - (1.0 / n) * np.sum((2 * i - 1) * (np.log(U) + np.log(1 - U[::-1])))
    # Small-sample correction (D'Agostino & Stephens 1986, for normal fit)
    A2_star = A2 * (1 + 0.75 / n + 2.25 / n ** 2) if estimate and dist == "norm" else A2
    # Approximate p-value (Marsaglia & Marsaglia 2004 for A2*)
    if A2_star < 0.2:
        p = 1 - np.exp(-13.436 + 101.14 * A2_star - 223.73 * A2_star ** 2)
    elif A2_star < 0.34:
        p = 1 - np.exp(-8.318 + 42.796 * A2_star - 59.938 * A2_star ** 2)
    elif A2_star < 0.6:
        p = np.exp(0.9177 - 4.279 * A2_star - 1.38 * A2_star ** 2)
    else:
        p = np.exp(1.2937 - 5.709 * A2_star + 0.0186 * A2_star ** 2)
    return {"A2": float(A2), "A2_star": float(A2_star), "p": float(np.clip(p, 0, 1))}


if __name__ == "__main__":
    print("=== Anderson-Darling test (Anderson-Darling 1954) ===\n")
    rng = np.random.default_rng(0)

    scenarios = [
        ("N(0, 1)                 ", rng.normal(size=500)),
        ("t3 (heavy tail)         ", rng.standard_t(3, size=500)),
        ("Uniform on [-1, 1]      ", rng.uniform(-1, 1, size=500)),
        ("N(0, 1) + 2 %  outliers", np.concatenate([rng.normal(size=490), 8 * rng.normal(size=10)])),
    ]
    print(f"  Testing goodness-of-fit to Normal (params estimated):\n")
    for name, x in scenarios:
        r = anderson_darling(x, dist="norm", estimate=True)
        print(f"  {name}  A^2* = {r['A2_star']:.4f}   p = {r['p']:.4f}")

    print("\n  A-D vs KS: A-D emphasises tails and typically has higher power")
    print("  for heavy-tail alternatives (t_3, outliers).")

    print("\n--- library cross-check (nortest R; scipy.stats.anderson Python) ---")

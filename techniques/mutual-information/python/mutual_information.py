"""Mutual information as an association measure (Reference §4.14).

For two discrete variables X and Y with joint distribution p(x, y) and
marginals p(x), p(y):

    I(X; Y) = sum_{x, y}  p(x, y) * log( p(x, y) / (p(x) p(y)) )

Units depend on the log base: bits (log2), nats (ln), Hartleys (log10).

Properties
----------
- I(X; Y) >= 0; I = 0 iff X and Y are independent.
- Symmetric: I(X; Y) = I(Y; X).
- Captures ANY dependence (linear, nonlinear, non-monotonic).
- Unbounded above; the *normalized* versions live in [0, 1]:
    NMI_arith = I / ((H(X) + H(Y)) / 2)
    NMI_geom  = I / sqrt(H(X) * H(Y))
    NMI_max   = I / max(H(X), H(Y))
  where H(X) = -sum p(x) log p(x) is the entropy of X.

Continuous variables
--------------------
For continuous data we estimate I via binning (Freedman-Diaconis bin width by
default) and use the discrete formula. This is biased for small n; better
estimators (KSG, kernel) exist (sklearn's mutual_info_regression uses k-NN).

Maximal Information Coefficient (MIC)
-------------------------------------
MIC searches over many grids and returns the maximum normalized mutual
information, rescaled to [0, 1]. Not implemented here -- use the ``minepy``
package for it; we note it in the README.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np


def entropy_discrete(x: Sequence, base: float = 2.0) -> float:
    """Shannon entropy H(X) from a discrete sample."""
    n = len(x)
    counts = Counter(x)
    return -sum((c / n) * math.log(c / n, base) for c in counts.values() if c > 0)


def mutual_information_discrete(x: Sequence, y: Sequence, base: float = 2.0) -> dict:
    """I(X; Y) and its normalized variants for two discrete (categorical) samples."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    joint = Counter(zip(x, y))
    px = Counter(x); py = Counter(y)
    I = 0.0
    for (xi, yi), c in joint.items():
        pxy = c / n; pxi = px[xi] / n; pyi = py[yi] / n
        I += pxy * math.log(pxy / (pxi * pyi), base)
    Hx = entropy_discrete(x, base)
    Hy = entropy_discrete(y, base)
    return {"I_xy": I, "H_x": Hx, "H_y": Hy,
            "nmi_arith": I / ((Hx + Hy) / 2) if (Hx + Hy) > 0 else 0.0,
            "nmi_geom": I / math.sqrt(Hx * Hy) if Hx > 0 and Hy > 0 else 0.0,
            "nmi_max": I / max(Hx, Hy) if max(Hx, Hy) > 0 else 0.0,
            "base": base}


def mutual_information_binned(x, y, bins=None, base: float = 2.0) -> dict:
    """MI for continuous x, y by binning each into ``bins`` (or Freedman-Diaconis).

    Crude but standard for a first pass. For better continuous estimators, see
    sklearn.feature_selection.mutual_info_regression (KSG / k-NN based).
    """
    x = np.asarray(x); y = np.asarray(y)
    if bins is None:
        def fd(a):
            q1, q3 = np.quantile(a, [0.25, 0.75])
            h = 2 * (q3 - q1) * len(a) ** (-1 / 3) or (a.max() - a.min()) / 10 or 1.0
            return max(2, int(math.ceil((a.max() - a.min()) / h)))
        bx, by = fd(x), fd(y)
    else:
        bx = by = bins
    xb = np.digitize(x, np.linspace(x.min(), x.max(), bx + 1)[1:-1])
    yb = np.digitize(y, np.linspace(y.min(), y.max(), by + 1)[1:-1])
    out = mutual_information_discrete(xb.tolist(), yb.tolist(), base=base)
    out["bins_x"] = bx; out["bins_y"] = by
    return out


def maximal_information_coefficient(x, y, alpha: float = 0.6, c: float = 15.0) -> dict:
    """Maximal Information Coefficient (Reshef et al. 2011).

    MIC searches over many (bx, by) bin partitions and returns the maximum
    NORMALIZED mutual information:
        MIC = max_{bx * by < B(n)}  I(X; Y; bx, by) / log2( min(bx, by) )
    with B(n) = n^alpha. The intuition: try lots of grids; report the best
    normalized score. Designed to be "equitable" across functional forms --
    a noisy line, a parabola, and a sinusoid of the same noise level should
    give similar MIC.

    Parameters
    ----------
    x, y  : continuous numeric vectors.
    alpha : exponent in the grid budget B(n) = n^alpha (default 0.6, per
            the original paper).
    c     : not used here (kept for API similarity to `minepy`).

    This is a simple grid-search reference implementation. The original paper's
    approximation algorithm (ApproxMaxMI) is faster; install ``minepy`` for it.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    B = int(max(4, n ** alpha))
    best = 0.0; best_grid = (2, 2)
    for bx in range(2, B + 1):
        for by in range(2, B + 1):
            if bx * by > B: continue
            xb = np.digitize(x, np.linspace(x.min(), x.max(), bx + 1)[1:-1])
            yb = np.digitize(y, np.linspace(y.min(), y.max(), by + 1)[1:-1])
            mi = mutual_information_discrete(xb.tolist(), yb.tolist(), base=2.0)["I_xy"]
            norm = math.log2(min(bx, by))
            if norm > 0:
                val = mi / norm
                if val > best:
                    best = val; best_grid = (bx, by)
    return {"MIC": min(1.0, best), "grid": best_grid, "B": B, "n": n}


def library_versions(x, y):
    from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
    # sklearn versions use natural log
    return {"sklearn mutual_info_score (nats)": float(mutual_info_score(x, y)),
            "sklearn normalized_mutual_info_score":
                float(normalized_mutual_info_score(x, y))}


if __name__ == "__main__":
    rng = np.random.default_rng(13)

    print("=== Discrete: two correlated categorical variables ===")
    # X in {0,1,2}; Y = X with 30% noise (random pick)
    x = rng.integers(0, 3, 1000)
    y = np.where(rng.random(1000) < 0.7, x, rng.integers(0, 3, 1000))
    for k, v in mutual_information_discrete(x.tolist(), y.tolist()).items():
        print(f"  {k:14s}: {v}")

    print("\n=== Continuous (binned): y = x^2 + noise ===")
    xc = rng.normal(0, 1, 500)
    yc = xc ** 2 + rng.normal(0, 0.2, 500)
    for k, v in mutual_information_binned(xc, yc).items():
        print(f"  {k:14s}: {v}")
    print(f"  Pearson r = {np.corrcoef(xc, yc)[0,1]:.4f}  (linear-only measure)")

    print("\n=== MIC (Maximal Information Coefficient) on the same continuous data ===")
    mic_res = maximal_information_coefficient(xc, yc, alpha=0.6)
    for k, v in mic_res.items():
        print(f"  {k:14s}: {v}")
    print("  (MIC is substantial despite Pearson r ~ 0; tighter values usually need")
    print("   the paper's optimization heuristic -- install 'minepy' for that.)")

    print("\n--- library (sklearn) ---")
    for k, v in library_versions(x.tolist(), y.tolist()).items():
        print(f"  {k}: {v}")

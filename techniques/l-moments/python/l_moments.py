"""L-moments and probability-weighted moments (Reference §1.24).

L-moments are linear combinations of order statistics that act like conventional
moments but are far more robust and *always exist* (even for heavy-tailed
distributions with infinite variance). Widely used in hydrology / extreme-value
analysis for fitting flood return periods, and good for small samples.

Probability-weighted moments (PWMs), unbiased sample estimators (Landwehr et al.):
  b_r = (1/n) * sum_{i=1}^{n}  [ C(i-1, r) / C(n-1, r) ] * x_(i)      (x sorted ascending)
  so  b_0 = mean.

L-moments from PWMs:
  L1 = b0
  L2 = 2 b1 - b0
  L3 = 6 b2 - 6 b1 + b0
  L4 = 20 b3 - 30 b2 + 12 b1 - b0

L-moment ratios:
  L-CV   tau   = L2 / L1            (analogue of CV; for positive data)
  L-skewness tau3 = L3 / L2         in (-1, 1)
  L-kurtosis tau4 = L4 / L2         in (-1/4, 1) roughly
"""
from __future__ import annotations

import math
from typing import Sequence


def _comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k)


def pwm(x: Sequence[float], max_order: int = 3):
    """Unbiased sample probability-weighted moments b_0 .. b_{max_order}."""
    s = sorted(float(v) for v in x)
    n = len(s)
    if n <= max_order:
        raise ValueError(f"need n > {max_order} observations")
    bs = []
    for r in range(max_order + 1):
        denom = _comb(n - 1, r)
        total = sum(_comb(i - 1, r) * s[i - 1] for i in range(1, n + 1))
        bs.append(total / (n * denom))
    return bs  # [b0, b1, b2, b3]


def l_moments(x: Sequence[float]):
    """Return dict with L1..L4 and the ratios L-CV, L-skewness, L-kurtosis."""
    b0, b1, b2, b3 = pwm(x, max_order=3)
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return {
        "L1_location": l1,
        "L2_scale": l2,
        "L3": l3,
        "L4": l4,
        "L_CV": l2 / l1 if l1 != 0 else float("nan"),
        "L_skewness": l3 / l2 if l2 != 0 else float("nan"),
        "L_kurtosis": l4 / l2 if l2 != 0 else float("nan"),
    }


def library_versions(x: Sequence[float]):
    out = {}
    try:
        import lmoments3 as lm

        out["lmoments3.lmom_ratios (L1,L2,t3,t4)"] = lm.lmom_ratios(list(x), nmom=4)
    except ImportError:
        out["note"] = "install 'lmoments3' for a reference implementation (pip install lmoments3)"
    # scipy has no L-moments; the from-scratch values above are the reference here.
    return out


if __name__ == "__main__":
    import random

    random.seed(0)
    # right-skewed, heavy-tailed-ish sample (mimics annual flood maxima)
    data = sorted(round(random.lognormvariate(3.0, 0.7), 1) for _ in range(40))
    print(f"n = {len(data)}  (right-skewed sample)\n")
    print("--- probability-weighted moments ---")
    for r, b in enumerate(pwm(data)):
        print(f"b{r} = {b:.4f}")
    print("\n--- L-moments ---")
    for k, v in l_moments(data).items():
        print(f"{k:14s}: {v:.4f}")

    print("\nInterpretation: L_skewness > 0 -> right-skewed; compare with the conventional")
    print("(more outlier-sensitive) skewness in techniques/shape-skewness-kurtosis.")

    print("\n--- library ---")
    for k, v in library_versions(data).items():
        print(f"{k}: {v}")

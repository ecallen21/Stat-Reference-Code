"""Cliff's delta effect size (Reference §7.16).

Nonparametric effect size for two independent ordinal (or continuous)
samples X and Y:

    delta = Pr(X > Y) - Pr(X < Y)      in [-1, 1]

- delta = +1: every X > every Y (perfect separation)
- delta = -1: every X < every Y
- delta =  0: no stochastic dominance

Related to the Mann-Whitney U statistic:
    delta = 2 * (U / (n_X n_Y)) - 1
Equivalent to  A_ML - (1 - A_ML)  where A_ML is Vargha-Delaney A.

Interpretation rule of thumb (Romano et al. 2006):
    |delta| < 0.147  negligible
    0.147 - 0.33     small
    0.33 - 0.474     medium
    > 0.474          large

Asymptotic CI: Cliff (1993) SE from the ranks; bootstrap for small n.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def cliff_delta(x, y, alpha: float = 0.05) -> dict:
    """Cliff's delta with asymptotic Normal CI."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    # Pair-wise comparison
    n_greater = int(np.sum([np.sum(xi > y) for xi in x]))
    n_less = int(np.sum([np.sum(xi < y) for xi in x]))
    delta = (n_greater - n_less) / (nx * ny)
    # Variance estimator (Cliff 1993, without-ties simplified)
    # Compute per-i dominance counts d_i
    d_i = np.array([np.sum(xi > y) - np.sum(xi < y) for xi in x]) / ny
    d_j = np.array([np.sum(yj < x) - np.sum(yj > x) for yj in y]) / nx
    var = ((ny - 1) * np.var(d_i, ddof=1) + (nx - 1) * np.var(d_j, ddof=1) +
           np.var(np.concatenate([d_i, d_j]), ddof=1)) / (nx * ny)
    se = math.sqrt(max(var, 0))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"cliff_delta": float(delta), "SE": float(se),
            "ci_low": float(delta - z * se), "ci_high": float(delta + z * se),
            "n_greater": n_greater, "n_less": n_less,
            "magnitude": _magnitude(abs(delta)),
            "method": "Cliff's delta"}


def _magnitude(abs_d: float) -> str:
    if abs_d < 0.147: return "negligible"
    if abs_d < 0.33:  return "small"
    if abs_d < 0.474: return "medium"
    return "large"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 40)
    y = rng.normal(0.0, 1, 40)
    r = cliff_delta(x, y)
    print("=== Cliff's delta (x ~ N(0.5, 1), y ~ N(0, 1)) ===")
    print(f"  delta = {r['cliff_delta']:.4f}  ({r['magnitude']})")
    print(f"  SE = {r['SE']:.4f}   95% CI = ({r['ci_low']:.4f}, {r['ci_high']:.4f})")

    print("\n=== Large effect: x = N(2, 1), y = N(0, 1) ===")
    r = cliff_delta(rng.normal(2, 1, 50), rng.normal(0, 1, 50))
    print(f"  delta = {r['cliff_delta']:.4f}  ({r['magnitude']})")

    print("\n=== Null: two draws from N(0, 1) ===")
    r = cliff_delta(rng.normal(0, 1, 50), rng.normal(0, 1, 50))
    print(f"  delta = {r['cliff_delta']:.4f}  ({r['magnitude']})")

    print("\n--- library cross-check (R effsize::cliff.delta) ---")
    print("  R: effsize::cliff.delta(x, y)")

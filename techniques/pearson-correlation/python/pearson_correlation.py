"""Pearson product-moment correlation (Reference §4.1).

Measures the strength of a *linear* association between two continuous
variables.

    r = sum((x - x_bar)(y - y_bar)) / sqrt(sum((x - x_bar)^2) * sum((y - y_bar)^2))
      = cov(x, y) / (sd(x) * sd(y))

Range: -1 (perfect negative linear) ... 0 (no linear) ... +1 (perfect positive
linear). Important caveats:

  - r captures LINEAR association only; a strong nonlinear pattern can have r=0.
  - r is sensitive to outliers (sample variances/covariances are).
  - "Correlation does not imply causation."
  - The (x, y) pairs must be from a bivariate distribution where assumptions
    for the significance test (bivariate normal) hold; for ranks/heavy tails,
    prefer Spearman or Kendall.

Significance test (H0: rho = 0):
    t = r * sqrt(n - 2) / sqrt(1 - r^2)   ~  t_{n-2} under H0

CI for rho via the Fisher z transformation:
    z   = atanh(r),    SE(z) = 1 / sqrt(n - 3)
    z +/- z_alpha/2 * SE(z), then back-transform with tanh.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(x): return sum(x) / len(x)


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Pearson's r. ``x`` and ``y`` are paired samples of equal length."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    mx = _mean(x); my = _mean(y)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    return sxy / math.sqrt(sxx * syy)


def pearson_test(x: Sequence[float], y: Sequence[float],
                 alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Pearson correlation with t-test and Fisher-z CI.

    Returns r, t, df, p-value, and the 100*conf% CI for rho.
    """
    n = len(x)
    r = pearson_r(x, y)
    if abs(r) == 1.0:
        return {"r": r, "t": float("inf"), "df": n - 2, "p_value": 0.0,
                "ci_lower": r, "ci_upper": r,
                "method": "Pearson"}
    t = r * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    if alternative == "two-sided":
        p = 2 * stats.t.sf(abs(t), df)
    elif alternative == "greater":
        p = float(stats.t.sf(t, df))
    else:
        p = float(stats.t.cdf(t, df))
    # Fisher z CI
    z = math.atanh(r); se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(0.5 + conf / 2)
    lo = math.tanh(z - zc * se); hi = math.tanh(z + zc * se)
    return {"r": r, "t": t, "df": df, "p_value": float(p),
            "ci_lower": lo, "ci_upper": hi,
            "method": "Pearson", "alternative": alternative}


def library_versions(x, y):
    res = stats.pearsonr(x, y)
    return {
        "scipy.stats.pearsonr (statistic, pvalue)": (float(res.statistic), float(res.pvalue)),
        "scipy.stats.pearsonr CI (95%)": tuple(map(float, res.confidence_interval())),
    }


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(5)
    # Generate (x, y) with a true rho around 0.6
    n = 80
    x = rng.normal(0, 1, n)
    y = 0.6 * x + math.sqrt(1 - 0.6 ** 2) * rng.normal(0, 1, n)

    print(f"=== Pearson on n = {n} ===")
    for k, v in pearson_test(x.tolist(), y.tolist()).items():
        print(f"  {k:10s}: {v}")

    print("\n--- library (scipy.stats.pearsonr) ---")
    for k, v in library_versions(x, y).items():
        print(f"  {k}: {v}")

    # Demo of zero correlation despite a strong nonlinear pattern
    print("\n=== Zero r despite a (quadratic) relationship ===")
    x2 = np.linspace(-3, 3, 100)
    y2 = x2 ** 2 + rng.normal(0, 0.1, 100)
    print(f"  r = {pearson_r(x2.tolist(), y2.tolist()):.4f}  (visual: clear U-shape)")

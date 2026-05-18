"""Lin's concordance correlation coefficient (Reference §4.9).

Measures agreement between two *paired* continuous measurements (think:
two devices, two raters) -- not just linear correlation. Two perfectly
parallel measurements that differ by a constant offset have Pearson r = 1
but Lin's CCC < 1 (because the offset breaks AGREEMENT).

Definition (Lin 1989)
    rho_c = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

Equivalent factored form:
    rho_c = r * C_b
where r = Pearson correlation (the "precision" term)
and   C_b = "bias correction factor" =
        2 / (v + 1/v + u^2)
        v = SD(x) / SD(y)        (scale shift)
        u = (mean(x) - mean(y)) / sqrt(SD(x) * SD(y))  (location shift)

So Lin's CCC = (precision) * (accuracy). CCC = 1 iff x_i == y_i for all i
(perfect agreement, not just linear association).

Fisher z CI is the same form as Pearson's: z = atanh(rho_c), SE = 1/sqrt(n-3).
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _mean(x): return sum(x) / len(x)


def lins_ccc(x: Sequence[float], y: Sequence[float], conf: float = 0.95) -> dict:
    """Lin's concordance correlation with the precision/accuracy decomposition.

    Note: variances/covariances use the population denominator (n) for
    consistency with Lin's original formulas; the conventional 95% CI uses
    Fisher z on rho_c.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x)
    mx, my = _mean(x), _mean(y)
    vx = sum((xi - mx) ** 2 for xi in x) / n
    vy = sum((yi - my) ** 2 for yi in y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
    rho_c = (2 * cov) / (vx + vy + (mx - my) ** 2)

    sx, sy = math.sqrt(vx), math.sqrt(vy)
    pearson = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")
    v = sx / sy if sy > 0 else float("nan")
    u = (mx - my) / math.sqrt(sx * sy) if sx > 0 and sy > 0 else float("nan")
    C_b = 2 / (v + 1 / v + u ** 2) if sy > 0 and sx > 0 else float("nan")

    z = math.atanh(rho_c) if abs(rho_c) < 1 else math.copysign(20.0, rho_c)
    se = 1 / math.sqrt(n - 3)
    zc = stats.norm.ppf(0.5 + conf / 2)
    ci_lo = math.tanh(z - zc * se); ci_hi = math.tanh(z + zc * se)
    return {"ccc": rho_c, "pearson_r_precision": pearson, "C_b_accuracy": C_b,
            "scale_shift_v": v, "location_shift_u": u,
            "ci_lower": ci_lo, "ci_upper": ci_hi, "n": n}


def library_versions(x, y):
    try:
        import pingouin as pg
        import pandas as pd
        return {"pingouin.intraclass_corr (substitute - shows ICC not CCC)":
                "pingouin has no direct CCC; use Lin's formula or DescTools::CCC in R"}
    except ImportError:
        return {}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(12)

    print("=== High correlation, large bias  -> Pearson high, CCC low ===")
    x = rng.normal(50, 10, 60)
    y = x + 5 + rng.normal(0, 1, 60)          # nearly parallel, offset of +5
    for k, v in lins_ccc(x.tolist(), y.tolist()).items():
        print(f"  {k:24s}: {v}")
    pearson = float(np.corrcoef(x, y)[0, 1])
    print(f"  (Pearson r for reference) : {pearson:.4f}")

    print("\n=== Perfect agreement (y = x) ===")
    print(lins_ccc(x.tolist(), x.tolist()))

    print("\n=== Same correlation, scale shift  (y = 2 x + noise) ===")
    y2 = 2 * x + rng.normal(0, 1, 60)
    for k, v in lins_ccc(x.tolist(), y2.tolist()).items():
        print(f"  {k:24s}: {v}")

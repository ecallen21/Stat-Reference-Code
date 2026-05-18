"""One- and two-sample t-tests (Reference §3.4).

From-scratch implementations plus scipy equivalents.

Tests covered
-------------
- one-sample t-test         : H0: mean(x) == mu0
- two-sample independent
    - Student's t           : assumes equal variances; pools the SD
    - Welch's t             : unequal variances; Satterthwaite df (the modern default)
- paired t-test             : one-sample t-test on the pairwise differences

Output (dict)
  t, df, p_value, mean / mean_diff, se, ci_lower, ci_upper, cohens_d (when relevant)

Sign convention for two-sample tests: ``mean(x1) - mean(x2)``.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _mean(x):
    return sum(x) / len(x)


def _var(x, ddof=1):
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - ddof)


def _t_ci(diff: float, se: float, df: float, conf: float):
    """Two-sided t-CI: diff +/- t_{1 - alpha/2, df} * se."""
    tcrit = stats.t.ppf(0.5 + conf / 2, df)
    return diff - tcrit * se, diff + tcrit * se


def one_sample_t(x: Sequence[float], mu0: float = 0.0, alternative: str = "two-sided",
                 conf: float = 0.95) -> dict:
    """One-sample t-test: H0 mean(x) == mu0.

    Parameters
    ----------
    x : numeric sample.
    mu0 : null hypothesis value for the mean.
    alternative : "two-sided" / "less" / "greater".
    conf : confidence level for the returned CI on the mean.
    """
    n = len(x)
    m = _mean(x)
    s = math.sqrt(_var(x))
    se = s / math.sqrt(n)
    t = (m - mu0) / se
    df = n - 1
    p = _t_pvalue(t, df, alternative)
    lo, hi = _t_ci(m, se, df, conf)
    return {"mean": m, "se": se, "t": t, "df": df, "p_value": p,
            "ci_lower": lo, "ci_upper": hi}


def two_sample_t(x1: Sequence[float], x2: Sequence[float], equal_var: bool = False,
                 alternative: str = "two-sided", conf: float = 0.95) -> dict:
    """Two-sample independent t-test.

    Parameters
    ----------
    x1, x2 : the two independent samples.
    equal_var : ``True`` -> Student's pooled-variance test (assumes equal variances);
        ``False`` -> Welch's test with Satterthwaite df (recommended default).
    alternative, conf : as in ``one_sample_t``.

    Sign convention: ``mean(x1) - mean(x2)``.
    """
    n1, n2 = len(x1), len(x2)
    m1, m2 = _mean(x1), _mean(x2)
    v1, v2 = _var(x1), _var(x2)
    diff = m1 - m2
    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)  # pooled variance
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
        d = diff / math.sqrt(sp2)                              # Cohen's d (pooled)
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )  # Welch-Satterthwaite df
        d = diff / math.sqrt((v1 + v2) / 2)                    # rough d for unequal var
    t = diff / se
    p = _t_pvalue(t, df, alternative)
    lo, hi = _t_ci(diff, se, df, conf)
    return {"mean_diff": diff, "se": se, "t": t, "df": df, "p_value": p,
            "ci_lower": lo, "ci_upper": hi, "cohens_d": d,
            "method": "Student" if equal_var else "Welch"}


def paired_t(x1: Sequence[float], x2: Sequence[float], alternative: str = "two-sided",
             conf: float = 0.95) -> dict:
    """Paired t-test = one-sample t-test on the differences ``x1 - x2``.

    Use when each x1[i] is paired with x2[i] (pre/post, twin pairs, matched cases).
    """
    if len(x1) != len(x2):
        raise ValueError("paired test requires equal-length samples")
    d = [a - b for a, b in zip(x1, x2)]
    res = one_sample_t(d, mu0=0.0, alternative=alternative, conf=conf)
    res["mean_diff"] = res.pop("mean")
    return res


def _t_pvalue(t: float, df: float, alternative: str) -> float:
    if alternative == "two-sided":
        return 2 * stats.t.sf(abs(t), df)
    if alternative == "greater":
        return float(stats.t.sf(t, df))
    if alternative == "less":
        return float(stats.t.cdf(t, df))
    raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")


def library_versions(x1, x2, mu0=100.0, paired_a=None, paired_b=None):
    """scipy.stats equivalents to cross-check."""
    out = {
        "one-sample (scipy)": stats.ttest_1samp(x1, popmean=mu0),
        "two-sample Student (scipy)": stats.ttest_ind(x1, x2, equal_var=True),
        "two-sample Welch (scipy)": stats.ttest_ind(x1, x2, equal_var=False),
    }
    if paired_a is not None and paired_b is not None and len(paired_a) == len(paired_b):
        out["paired (scipy)"] = stats.ttest_rel(paired_a, paired_b)
    return out


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(7)
    a = rng.normal(105, 14, 30).tolist()
    b = rng.normal(100, 18, 27).tolist()
    pre = rng.normal(50, 8, 20).tolist()
    post = [p + rng.normal(2, 3) for p in pre]

    print("=== one-sample t (H0: mean = 100) ===")
    for k, v in one_sample_t(a, mu0=100).items():
        print(f"  {k:10s}: {v}")

    print("\n=== two-sample Welch (a vs b) ===")
    for k, v in two_sample_t(a, b, equal_var=False).items():
        print(f"  {k:10s}: {v}")

    print("\n=== two-sample Student (a vs b) ===")
    for k, v in two_sample_t(a, b, equal_var=True).items():
        print(f"  {k:10s}: {v}")

    print("\n=== paired (post vs pre) ===")
    for k, v in paired_t(post, pre).items():
        print(f"  {k:10s}: {v}")

    print("\n--- library (scipy) ---")
    for k, v in library_versions(a, b, mu0=100, paired_a=post, paired_b=pre).items():
        print(f"  {k}: {v}")

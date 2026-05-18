"""Intraclass correlation coefficient (ICC) (Reference §4.6).

ICC quantifies the proportion of total variance attributable to the
*between-subject* differences vs. the *within-subject* (measurement) noise.
The standard use case: ``n`` subjects each rated by ``k`` raters; how
consistent are the raters?

Shrout-Fleiss (1979) classification -- six variants depending on the design
and the unit you'll report:

  ICC model  | k raters drawn from a    | single-measure (1,1)/(2,1)/(3,1) |  average-of-k  (1,k)/(2,k)/(3,k)
  -----------|--------------------------|---------------------------------|------------------------------------
  ICC1       | population (one-way)     | ICC(1, 1) -- raters are random  | ICC(1, k)
  ICC2       | sample (two-way random)  | ICC(2, 1) -- raters are random  | ICC(2, k)   (a.k.a. ICC(A,1) / ICC(A,k))
  ICC3       | the only raters of int.  | ICC(3, 1) -- raters fixed       | ICC(3, k)   (a.k.a. ICC(C,1) / ICC(C,k))

McGraw & Wong (1996) variants -- absolute agreement (A) vs. consistency (C).
We expose ICC(A, 1), ICC(A, k), ICC(C, 1), ICC(C, k) as the modern way to ask
the question.

Mean squares (from a two-way ANOVA: subjects x raters, no interaction)
  MS_R   row mean square (subjects)
  MS_C   column mean square (raters)
  MS_E   residual mean square

Formulas (n subjects, k raters)
  ICC(1, 1) = (MS_B - MS_W) / (MS_B + (k - 1) * MS_W)              one-way ANOVA
  ICC(A, 1) = (MS_R - MS_E) / (MS_R + (k - 1) * MS_E + k(MS_C - MS_E)/n)
  ICC(A, k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)
  ICC(C, 1) = (MS_R - MS_E) / (MS_R + (k - 1) * MS_E)
  ICC(C, k) = (MS_R - MS_E) / MS_R

Inputs: ``data`` is a 2D array shape (n subjects, k raters).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import NamedTuple, Sequence    # stdlib: type hints (Sequence = indexable iterable; NamedTuple = tuple with named fields)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


class TwoWayMS(NamedTuple):
    """Mean-square decomposition for a two-way subjects-by-raters ANOVA.

    Unpacks like a 5-tuple, so ``MSr, MSc, MSe, n, k = _two_way_anova_ms(data)`` still works.
    """
    MS_rows: float      # mean square between subjects (rows)
    MS_cols: float      # mean square between raters (cols)
    MS_error: float     # residual mean square
    n: int              # number of subjects
    k: int              # number of raters


def _two_way_anova_ms(data: np.ndarray):
    """Return a TwoWayMS named tuple for an n x k subjects-by-raters table."""
    n, k = data.shape
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_r = k * np.sum((row_means - grand) ** 2)
    ss_c = n * np.sum((col_means - grand) ** 2)
    ss_t = np.sum((data - grand) ** 2)
    ss_e = ss_t - ss_r - ss_c
    df_r, df_c, df_e = n - 1, k - 1, (n - 1) * (k - 1)
    return TwoWayMS(MS_rows=ss_r / df_r, MS_cols=ss_c / df_c,
                    MS_error=ss_e / df_e, n=n, k=k)


def icc(data, model: str = "two-way", form: str = "absolute",
        unit: str = "single") -> dict:
    """Compute one of the six common ICC variants.

    Parameters
    ----------
    data : array-like, shape (n_subjects, k_raters).
    model : "one-way" (ICC1) or "two-way" (ICC2/ICC3).
        ICC2: raters are a random sample of all possible raters.
        ICC3: the raters in the study are the only raters of interest.
    form : "absolute" (ICC(A, *)) -- agreement across raters in raw value;
           "consistency" (ICC(C, *)) -- only relative ordering matters.
        For one-way, form is fixed at "absolute".
    unit : "single" -- reliability of a single rater's score;
           "average" -- reliability of the mean of all k raters.
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape

    if model == "one-way":
        # MS_B = between-row, MS_W = within-row (averaged across columns)
        row_means = data.mean(axis=1); grand = data.mean()
        ss_b = k * np.sum((row_means - grand) ** 2)
        ss_w = np.sum((data - row_means[:, None]) ** 2)
        ms_b = ss_b / (n - 1); ms_w = ss_w / (n * (k - 1))
        icc1_1 = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
        icc1_k = (ms_b - ms_w) / ms_b
        return {"variant": "ICC(1, 1)" if unit == "single" else "ICC(1, k)",
                "value": icc1_1 if unit == "single" else icc1_k,
                "MS_B": ms_b, "MS_W": ms_w, "n": n, "k": k}

    ms_r, ms_c, ms_e, n_, k_ = _two_way_anova_ms(data)
    if form == "absolute":
        if unit == "single":
            val = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
            label = "ICC(A, 1)"
        else:
            val = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
            label = "ICC(A, k)"
    elif form == "consistency":
        if unit == "single":
            val = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e)
            label = "ICC(C, 1)"
        else:
            val = (ms_r - ms_e) / ms_r
            label = "ICC(C, k)"
    else:
        raise ValueError("form must be 'absolute' or 'consistency'")
    return {"variant": label, "value": val,
            "MS_R": ms_r, "MS_C": ms_c, "MS_E": ms_e, "n": n, "k": k}


def library_versions(data):
    try:
        import pingouin as pg
        import pandas as pd
        n, k = np.shape(data)
        long = pd.DataFrame({
            "subject": np.repeat(np.arange(n), k),
            "rater": np.tile(np.arange(k), n),
            "score": np.asarray(data).reshape(-1),
        })
        return {"pingouin.intraclass_corr":
                pg.intraclass_corr(data=long, targets="subject", raters="rater",
                                   ratings="score")}
    except ImportError:
        return {"note": "install 'pingouin' for pg.intraclass_corr"}


if __name__ == "__main__":
    rng = np.random.default_rng(10)
    n, k = 30, 4
    true_score = rng.normal(50, 10, n)
    rater_bias = rng.normal(0, 1.5, k)         # raters differ in level (absolute)
    noise = rng.normal(0, 3, (n, k))
    data = true_score[:, None] + rater_bias[None, :] + noise

    print("Variant       | value")
    for model, form, unit in [
        ("one-way", "absolute", "single"), ("one-way", "absolute", "average"),
        ("two-way", "absolute", "single"), ("two-way", "absolute", "average"),
        ("two-way", "consistency", "single"), ("two-way", "consistency", "average"),
    ]:
        r = icc(data, model=model, form=form, unit=unit)
        print(f"  {r['variant']:12s}: {r['value']:+.4f}")

    print("\n--- library ---")
    for k_, v in library_versions(data).items():
        print(f"  {k_}:\n{v}")

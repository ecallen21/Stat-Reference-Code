"""Categorical variable coding schemes (Reference §5.32).

A categorical predictor with k levels needs to be turned into k - 1 columns of
the design matrix. The choice of coding doesn't affect the FITTED VALUES (or
R^2 or sigma_hat), but it changes what the INDIVIDUAL COEFFICIENTS mean.

Four schemes implemented
------------------------
- Dummy (treatment) coding   : default in R / pandas
        col_j = 1 if obs is in level j (j != reference), else 0
        beta_j = mean(level_j) - mean(reference_level)

- Effect (sum-to-zero) coding
        col_j = 1 if level j, -1 if reference level, else 0
        beta_j = mean(level_j) - grand_mean
        (the reference level's deviation from the grand mean is -sum of the others)

- Helmert coding              : successive contrasts
        col_j compares level (j + 1) to the AVERAGE of levels 1..j
        Useful for ordered categories with theory-driven sequencing.

- Deviation coding           : each level vs grand mean (full k-1 set without a
                                "reference")
        col_j = 1 if level j, -1 if the *last* level, else 0
        Variant of effect coding where the omitted level is the last one.

The intercept's meaning also changes:
        dummy      : intercept = mean of the reference level
        effect     : intercept = grand mean
        Helmert    : intercept = mean of level 1
        deviation  : intercept = grand mean

This file builds each design and fits a one-factor ANOVA; the fitted values
are byte-for-byte identical across the four parameterizations.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _factor_levels(group):
    return list(sorted(set(group)))


def dummy_coding(group, reference=None):
    """[0/1] columns for each non-reference level (R's default contr.treatment)."""
    levels = _factor_levels(group)
    if reference is None:
        reference = levels[0]
    others = [lev for lev in levels if lev != reference]
    cols = []
    for lev in others:
        cols.append(np.array([1.0 if g == lev else 0.0 for g in group]))
    X = np.column_stack(cols) if cols else np.zeros((len(group), 0))
    names = [f"{lev}_vs_{reference}" for lev in others]
    return X, names, {"reference": reference, "scheme": "dummy"}


def effect_coding(group, reference=None):
    """[-1, +1] columns; reference is -1 in every column (R's contr.sum)."""
    levels = _factor_levels(group)
    if reference is None:
        reference = levels[-1]
    others = [lev for lev in levels if lev != reference]
    cols = []
    for lev in others:
        cols.append(np.array([1.0 if g == lev else (-1.0 if g == reference else 0.0)
                              for g in group]))
    X = np.column_stack(cols) if cols else np.zeros((len(group), 0))
    names = [f"{lev}_vs_grandmean" for lev in others]
    return X, names, {"reference": reference, "scheme": "effect (sum-to-zero)"}


def helmert_coding(group, levels: Sequence | None = None):
    """Successive contrasts: column j compares level (j+1) to the mean of levels 1..j."""
    if levels is None:
        levels = _factor_levels(group)
    levels = list(levels); k = len(levels)
    cols = []; names = []
    for j in range(1, k):
        col = np.zeros(len(group), dtype=float)
        # level j+1 (index j here) gets coefficient j; preceding levels each get -1
        for i, g in enumerate(group):
            if g == levels[j]:
                col[i] = j
            elif g in levels[:j]:
                col[i] = -1
        # scale so contrasts are orthogonal in design
        col = col / (j + 1)
        cols.append(col)
        names.append(f"{levels[j]}_vs_avg_of_prev")
    X = np.column_stack(cols) if cols else np.zeros((len(group), 0))
    return X, names, {"levels_order": levels, "scheme": "Helmert"}


def deviation_coding(group, omitted=None):
    """Each level coded +1 in its own col; the omitted level is -1 in every col.

    Conventionally distinct from 'effect' coding only in WHICH level is omitted
    (last vs. first); the math is the same family.
    """
    levels = _factor_levels(group)
    if omitted is None:
        omitted = levels[-1]
    others = [lev for lev in levels if lev != omitted]
    cols = []
    for lev in others:
        cols.append(np.array([1.0 if g == lev else (-1.0 if g == omitted else 0.0)
                              for g in group]))
    X = np.column_stack(cols) if cols else np.zeros((len(group), 0))
    names = [f"{lev}_vs_grandmean" for lev in others]
    return X, names, {"omitted": omitted, "scheme": "deviation"}


def fit_one_factor(group, y, coding_fn):
    """Fit y = intercept + factor(group) via OLS using ``coding_fn`` to build the design."""
    X_factor, names, meta = coding_fn(group)
    X = np.column_stack([np.ones(len(group)), X_factor])
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    rss = float(((y - yhat) ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    return {"scheme": meta["scheme"], "meta": meta,
            "intercept": float(beta[0]),
            "coefficients": dict(zip(names, beta[1:].tolist())),
            "fitted_first_10": yhat[:10].tolist(),
            "rss": rss, "r_squared": 1 - rss / tss}


def library_versions(group, y):
    import pandas as pd, statsmodels.formula.api as smf
    df = pd.DataFrame({"y": y, "g": pd.Categorical(group)})
    return {
        "R-style 'dummy' (Treatment)":
            smf.ols("y ~ C(g, Treatment)", data=df).fit().params.to_dict(),
        "R-style 'effect' (Sum)":
            smf.ols("y ~ C(g, Sum)", data=df).fit().params.to_dict(),
        "R-style 'Helmert'":
            smf.ols("y ~ C(g, Helmert)", data=df).fit().params.to_dict(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n_per = 30
    group_means = {"A": 50.0, "B": 55.0, "C": 60.0, "D": 52.0}
    grand_mean = sum(group_means.values()) / len(group_means)
    group = sum([[g] * n_per for g in group_means], [])
    y = np.concatenate([rng.normal(m, 3, n_per) for m in group_means.values()])

    print(f"True group means: {group_means}")
    print(f"True grand mean:  {grand_mean}\n")

    for fn in (dummy_coding, effect_coding, helmert_coding, deviation_coding):
        res = fit_one_factor(group, y, fn)
        print(f"=== {res['scheme']} ===")
        print(f"  intercept = {res['intercept']:.4f}")
        for nm, b in res["coefficients"].items():
            print(f"  {nm:24s}: {b:+.4f}")
        print(f"  R^2 = {res['r_squared']:.4f}    (identical across schemes)\n")

    # Sanity: the SAME fitted values across all four schemes
    yhats = []
    for fn in (dummy_coding, effect_coding, helmert_coding, deviation_coding):
        X, *_ = fn(group); X_full = np.column_stack([np.ones(len(group)), X])
        beta, *_ = np.linalg.lstsq(X_full, y, rcond=None)
        yhats.append(X_full @ beta)
    print("max |fitted_dummy - fitted_other| across schemes:")
    base = yhats[0]
    for fn_name, yh in zip(("effect", "Helmert", "deviation"), yhats[1:]):
        print(f"  {fn_name:10s}: {float(np.max(np.abs(yh - base))):.2e}")

    print("\n--- library (statsmodels with C(g, ...)) ---")
    for k, v in library_versions(group, y).items():
        print(f"  {k}: {v}")

"""Repeated-measures ANOVA + sphericity corrections (Reference §12.1).

For n subjects each measured under K within-subject conditions (or K time points):

    Total SS  =  SS_subjects  +  SS_condition  +  SS_error
    F = (SS_condition / (K - 1)) / (SS_error / ((n - 1)(K - 1)))
        ~ F(K - 1, (n - 1)(K - 1))    under H0: mu_1 = ... = mu_K
                                       AND sphericity of the within-subject
                                       covariance matrix.

Sphericity = equality of variances of ALL pairwise differences between conditions.
    Test with Mauchly's W (or Bartlett's test on differenced data).
    If violated, use one of two corrections to df:

Greenhouse-Geisser epsilon (GG):
    eps_GG = (tr(A))^2 / [(K - 1) * tr(A' A)]
    where A = C S C',  C = centering matrix,  S = within-subject cov matrix.
    Multiply BOTH df of F by eps_GG.

Huynh-Feldt epsilon (HF):
    eps_HF = min(1, (n(K - 1)*eps_GG - 2) / ((K - 1)(n - 1 - (K - 1)*eps_GG)))
    Less conservative than GG when true epsilon > 0.75.

Rules of thumb: if eps_GG < 0.75 use GG; if >= 0.75 use HF (Girden 1992).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def rm_anova(data) -> dict:
    """One-way repeated-measures ANOVA on an n x K matrix.

    Rows are subjects, columns are conditions/times. All subjects must have
    complete data (no missing) for this classic formulation.
    """
    Y = np.asarray(data, dtype=float)
    if Y.ndim != 2:
        raise ValueError("data must be a 2D n x K matrix (rows=subjects, cols=conditions)")
    n, K = Y.shape
    grand = Y.mean()
    subj = Y.mean(axis=1)              # per-subject mean, length n
    cond = Y.mean(axis=0)              # per-condition mean, length K

    # Sums of squares (Type III on balanced within-subjects design = these)
    SS_total     = float(((Y - grand) ** 2).sum())
    SS_subjects  = float(K * ((subj - grand) ** 2).sum())
    SS_condition = float(n * ((cond - grand) ** 2).sum())
    SS_error     = SS_total - SS_subjects - SS_condition

    df_cond = K - 1
    df_err  = (n - 1) * (K - 1)
    MS_cond = SS_condition / df_cond
    MS_err  = SS_error / df_err
    F = MS_cond / MS_err
    p = float(stats.f.sf(F, df_cond, df_err))

    # Greenhouse-Geisser and Huynh-Feldt sphericity corrections
    # Within-subject covariance matrix of the K columns, then center via C
    S = np.cov(Y.T, ddof=1)                          # K x K
    C = np.eye(K) - np.ones((K, K)) / K              # centering matrix
    A = C @ S @ C
    tr_A = float(np.trace(A))
    tr_A2 = float(np.trace(A @ A))
    eps_gg = (tr_A ** 2) / ((K - 1) * tr_A2) if tr_A2 > 0 else 1.0
    eps_gg = max(1.0 / (K - 1), min(1.0, eps_gg))     # bounded in [1/(K-1), 1]
    # Huynh-Feldt (Lecoutre 1991 corrected form)
    num = n * (K - 1) * eps_gg - 2
    den = (K - 1) * (n - 1 - (K - 1) * eps_gg)
    eps_hf = min(1.0, num / den) if den > 0 else 1.0

    p_gg = float(stats.f.sf(F, df_cond * eps_gg, df_err * eps_gg))
    p_hf = float(stats.f.sf(F, df_cond * eps_hf, df_err * eps_hf))

    # Partial eta-squared (effect size for the condition effect)
    eta2_partial = SS_condition / (SS_condition + SS_error)

    return {"n_subjects": n, "K_conditions": K,
            "SS_subjects": SS_subjects, "SS_condition": SS_condition,
            "SS_error": SS_error, "SS_total": SS_total,
            "df_condition": df_cond, "df_error": df_err,
            "MS_condition": MS_cond, "MS_error": MS_err,
            "F": F, "p_uncorrected": p,
            "epsilon_GreenhouseGeisser": eps_gg,
            "epsilon_HuynhFeldt": eps_hf,
            "p_GreenhouseGeisser": p_gg,
            "p_HuynhFeldt": p_hf,
            "partial_eta_squared": eta2_partial,
            "method": "one-way repeated-measures ANOVA + sphericity corrections"}


def library_versions(data):
    """Cross-check via statsmodels or pingouin if available."""
    try:
        import pingouin as pg
        import pandas as pd
        Y = np.asarray(data, dtype=float)
        n, K = Y.shape
        long = pd.DataFrame({
            "subject": np.repeat(np.arange(n), K),
            "condition": np.tile(np.arange(K), n),
            "y": Y.flatten()
        })
        r = pg.rm_anova(dv="y", within="condition", subject="subject",
                        data=long, correction=True)
        return {"pingouin.rm_anova": r.to_dict(orient="list")}
    except Exception as ex:
        return {"pingouin (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    # 30 subjects, 4 conditions; true condition means 0, 0.3, 0.6, 0.4
    n = 30; K = 4
    subj_effect = rng.normal(0, 0.7, n)              # subject-level heterogeneity
    cond_means = np.array([0.0, 0.3, 0.6, 0.4])
    Y = subj_effect[:, None] + cond_means[None, :] + rng.normal(0, 0.5, (n, K))

    print("=== Repeated-measures ANOVA (n=30, K=4) ===")
    r = rm_anova(Y)
    print(f"  F({r['df_condition']}, {r['df_error']}) = {r['F']:.4f}")
    print(f"  p (uncorrected)      = {r['p_uncorrected']:.4g}")
    print(f"  epsilon_GG           = {r['epsilon_GreenhouseGeisser']:.4f}")
    print(f"  epsilon_HF           = {r['epsilon_HuynhFeldt']:.4f}")
    print(f"  p (GG corrected)     = {r['p_GreenhouseGeisser']:.4g}")
    print(f"  p (HF corrected)     = {r['p_HuynhFeldt']:.4g}")
    print(f"  partial eta^2        = {r['partial_eta_squared']:.4f}")

    print("\n--- library (pingouin) ---")
    for k, v in library_versions(Y).items():
        print(f"  {k}: {v}")

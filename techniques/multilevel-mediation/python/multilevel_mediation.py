"""Multilevel 1-1-1 Mediation with within/between decomposition (Reference §12.22).

X (predictor), M (mediator), Y (outcome) all measured REPEATEDLY per subject.
Standard mediation:
    a: X  -> M
    b: M  -> Y  (controlling for X)
    c': X -> Y  (direct, controlling for M)
    ab: indirect effect

Multilevel twist: X and M vary WITHIN each subject and across subjects. Naive
per-observation mediation confounds the between-subject and within-subject
effects. Solution: group-mean-center X and M within each subject to obtain
WITHIN effects (a_w, b_w) and use the group means to get BETWEEN effects
(a_b, b_b). Indirect effect decomposes similarly:

    indirect_within  = a_w * b_w
    indirect_between = a_b * b_b

Bauer-Preacher-Gil (2006) framework.

Confidence interval on the indirect effect: MONTE CARLO CI (draw from the
joint sampling distribution of (a, b) and compute the empirical percentile CI
of a*b). More reliable than Sobel's product SE for small samples.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _group_mean(x, cluster):
    """Return per-observation vector of x's cluster mean."""
    means = {c: x[cluster == c].mean() for c in np.unique(cluster)}
    return np.array([means[c] for c in cluster])


def multilevel_111_mediation(X, M, Y, cluster, n_mc: int = 5000, seed: int = 0) -> dict:
    """1-1-1 multilevel mediation with within/between decomposition.

    Fits pooled OLS on person-centered / person-mean variables. Simplified
    from a full random-effect fit, adequate for the demo.
    """
    X = np.asarray(X, dtype=float); M = np.asarray(M, dtype=float)
    Y = np.asarray(Y, dtype=float); cluster = np.asarray(cluster)
    Xb = _group_mean(X, cluster); Mb = _group_mean(M, cluster)     # per-cluster means
    Xw = X - Xb; Mw = M - Mb                                        # within-cluster deviations

    n = len(Y); p = 5
    # Y = intercept + b_w * Mw + b_b * Mb + c'_w * Xw + c'_b * Xb
    D_y = np.column_stack([np.ones(n), Mw, Mb, Xw, Xb])
    beta_y, *_ = np.linalg.lstsq(D_y, Y, rcond=None)
    resid_y = Y - D_y @ beta_y
    sigma2_y = float((resid_y ** 2).sum() / (n - p))
    cov_y = sigma2_y * np.linalg.pinv(D_y.T @ D_y)
    b_w = beta_y[1]; b_b = beta_y[2]
    se_b_w = math.sqrt(cov_y[1, 1]); se_b_b = math.sqrt(cov_y[2, 2])

    # M = intercept + a_w * Xw + a_b * Xb
    D_m = np.column_stack([np.ones(n), Xw, Xb])
    beta_m, *_ = np.linalg.lstsq(D_m, M, rcond=None)
    resid_m = M - D_m @ beta_m
    sigma2_m = float((resid_m ** 2).sum() / (n - 3))
    cov_m = sigma2_m * np.linalg.pinv(D_m.T @ D_m)
    a_w = beta_m[1]; a_b = beta_m[2]
    se_a_w = math.sqrt(cov_m[1, 1]); se_a_b = math.sqrt(cov_m[2, 2])

    # Monte Carlo CI on the indirect effects
    rng = np.random.default_rng(seed)
    a_w_draws = rng.normal(a_w, se_a_w, n_mc); b_w_draws = rng.normal(b_w, se_b_w, n_mc)
    a_b_draws = rng.normal(a_b, se_a_b, n_mc); b_b_draws = rng.normal(b_b, se_b_b, n_mc)
    indirect_w_draws = a_w_draws * b_w_draws
    indirect_b_draws = a_b_draws * b_b_draws
    return {"within": {
                "a_w": float(a_w), "SE_a_w": float(se_a_w),
                "b_w": float(b_w), "SE_b_w": float(se_b_w),
                "indirect_a*b_w": float(a_w * b_w),
                "CI95_MC_within": {"lower": float(np.quantile(indirect_w_draws, 0.025)),
                                    "upper": float(np.quantile(indirect_w_draws, 0.975))}},
            "between": {
                "a_b": float(a_b), "SE_a_b": float(se_a_b),
                "b_b": float(b_b), "SE_b_b": float(se_b_b),
                "indirect_a*b_b": float(a_b * b_b),
                "CI95_MC_between": {"lower": float(np.quantile(indirect_b_draws, 0.025)),
                                     "upper": float(np.quantile(indirect_b_draws, 0.975))}},
            "direct_within c'_w": float(beta_y[3]),
            "direct_between c'_b": float(beta_y[4]),
            "n": int(n), "n_clusters": int(len(np.unique(cluster))), "n_mc": n_mc,
            "note": ("Person-mean-centered OLS approximation of Bauer-Preacher-Gil "
                     "(2006). A full multilevel SEM fit (lavaan, blavaan) treats "
                     "the group means as latent variables with their own SE."),
            "method": "1-1-1 multilevel mediation (within/between decomposition + MC CI)"}


if __name__ == "__main__":
    rng = np.random.default_rng(53)
    n_subj = 60; n_time = 8; n = n_subj * n_time
    cluster = np.repeat(np.arange(n_subj), n_time)
    # true within-person effects: a_w = 0.4, b_w = 0.5 -> indirect_w = 0.2
    # true between-person effects: a_b = 0.3, b_b = 0.6 -> indirect_b = 0.18
    X_between = rng.normal(0, 1, n_subj)
    X_within = rng.normal(0, 1, n)
    X = X_within + X_between[cluster]
    # M = a_w * X_w + a_b * X_b + noise    (a_w = 0.4, a_b = 0.3)
    M = 0.4 * X_within + 0.3 * X_between[cluster] + rng.normal(0, 0.5, n)
    M_within = M - _group_mean(M, cluster)
    M_between = _group_mean(M, cluster)
    # Y = b_w * M_w + b_b * M_b + c'_w * X_w + c'_b * X_b + noise
    # true b_w = 0.5, b_b = 0.6, c'_w = 0.1, c'_b = 0.1
    Y = 0.5 * M_within + 0.6 * M_between + 0.1 * X_within + 0.1 * X_between[cluster] \
        + rng.normal(0, 0.5, n)

    fit = multilevel_111_mediation(X, M, Y, cluster)
    print("=== Within-person effects (true a_w=0.4, b_w=0.5, indirect=0.20) ===")
    print(f"  {fit['within']}")
    print("\n=== Between-person effects (true a_b=0.3, b_b=0.6, indirect=0.18) ===")
    print(f"  {fit['between']}")
    print(f"\n=== Direct effects ===")
    key_w = "direct_within c'_w"; key_b = "direct_between c'_b"
    print(f"  c'_w = {fit[key_w]:+.4f}")
    print(f"  c'_b = {fit[key_b]:+.4f}")

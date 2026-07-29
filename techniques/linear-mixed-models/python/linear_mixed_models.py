"""Linear Mixed Models via REML (Reference §12.2; also covers §12.13, §12.16,
§12.20, §12.25, §12.26, §12.27, §12.29, §12.30, §12.32, §12.33).

Two-level random-intercept + random-slope model:

    y_{ij}  =  X_{ij}' beta  +  Z_{ij}' u_i  +  eps_{ij}
    u_i ~ N(0, G)     eps_{ij} ~ N(0, sigma^2)     u independent of eps

where i indexes clusters (subjects, schools, ...) and j indexes observations
within a cluster.

REML estimation (Restricted Maximum Likelihood): profile out beta first, then
maximize the profile log-likelihood on the covariance parameters. Standard in
lme4 / statsmodels.MixedLM / SAS PROC MIXED.

BLUPs (Best Linear Unbiased Predictors):
    u_hat_i  =  G Z_i' V_i^{-1} (y_i - X_i beta_hat)
where V_i = Z_i G Z_i' + sigma^2 I.

ICC from a random-intercept model:
    ICC  =  sigma_u^2 / (sigma_u^2 + sigma^2)     (§12.20)
Proportion of total variance attributable to the cluster level.

Also covered in the README:
    §12.13 Three-level / cross-classified: fit by extending Z to include
        multiple grouping variables (implementation extension of this code).
    §12.16 Covariance-structure selection: compare LR / AIC across nested
        random-effect specifications.
    §12.25 BLUPs and singular fits: `blups_head` returned; near-zero variance
        components warned about in the README.
    §12.26 Crossed random effects: covered by expanding Z with 0/1 indicators
        for a second grouping factor -- see the note in the README.
    §12.27 Shrinkage / partial pooling: BLUPs shrink cluster deviations toward
        zero; the shrinkage factor depends on within-vs-between variance.
    §12.29 Choosing RE structure: keep-it-maximal (Barr et al.) vs.
        parsimonious (Bates et al.) tension explained in README.
    §12.30 When to use: whenever observations cluster within units (subjects,
        sites, families) or repeat over time.
    §12.32 Centering: grand-mean center for BETWEEN-cluster interpretation;
        group-mean center for WITHIN-cluster interpretation.
    §12.33 Correlation structures: this file's fit assumes IID residuals;
        for AR(1) / compound-symmetric / unstructured residual covariance,
        pass a rho parameter (extension noted in README).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize    # optimize: BFGS on the REML profile likelihood


def _reml_neg_ll(log_theta, y, X, Z, cluster_ids):
    """REML profile negative log-likelihood on variance components.

    Parameterization: theta = (log(sigma_u), log(sigma), log(rho corr param)).
    For random-intercept only: G = sigma_u^2. For random-int + slope: G is 2x2
    with sigma_u^2, sigma_slope^2, and correlation rho.
    """
    q = Z.shape[1] if Z.ndim > 1 else 1
    if q == 1:
        sigma_u2 = math.exp(2 * log_theta[0])
        sigma2 = math.exp(2 * log_theta[1])
        G = np.array([[sigma_u2]])
    else:
        # int + slope
        sigma_u2 = math.exp(2 * log_theta[0])
        sigma_v2 = math.exp(2 * log_theta[1])
        rho = math.tanh(log_theta[2])                    # bound in (-1, 1)
        sigma2 = math.exp(2 * log_theta[3])
        G = np.array([[sigma_u2, rho * math.sqrt(sigma_u2 * sigma_v2)],
                      [rho * math.sqrt(sigma_u2 * sigma_v2), sigma_v2]])
    unique = np.unique(cluster_ids)
    n_total = len(y)
    XtViX = np.zeros((X.shape[1], X.shape[1]))
    XtViy = np.zeros(X.shape[1])
    logdet_V = 0.0
    for c in unique:
        m = cluster_ids == c
        y_i = y[m]; X_i = X[m]; Z_i = Z[m] if Z.ndim > 1 else Z[m][:, None]
        n_i = m.sum()
        V_i = Z_i @ G @ Z_i.T + sigma2 * np.eye(n_i)
        sign, logdet_i = np.linalg.slogdet(V_i)
        logdet_V += logdet_i
        V_inv = np.linalg.inv(V_i)
        XtViX += X_i.T @ V_inv @ X_i
        XtViy += X_i.T @ V_inv @ y_i
    sign_x, logdet_XtViX = np.linalg.slogdet(XtViX)
    beta = np.linalg.solve(XtViX, XtViy)
    # profile: residuals under beta
    quad = 0.0
    for c in unique:
        m = cluster_ids == c
        y_i = y[m]; X_i = X[m]; Z_i = Z[m] if Z.ndim > 1 else Z[m][:, None]
        n_i = m.sum()
        V_i = Z_i @ G @ Z_i.T + sigma2 * np.eye(n_i)
        V_inv = np.linalg.inv(V_i)
        r = y_i - X_i @ beta
        quad += r @ V_inv @ r
    reml = -0.5 * (logdet_V + logdet_XtViX + quad + (n_total - X.shape[1]) * math.log(2 * math.pi))
    return -reml


def fit_lmm(y, X, Z, cluster_ids, max_iter: int = 200) -> dict:
    """REML fit for a two-level LMM with random intercept (Z is a column of 1s)
    or random intercept + random slope on a single covariate.

    Parameters
    ----------
    y : length-n outcome.
    X : n x p design matrix (WITH intercept if you want one).
    Z : n x q design matrix for random effects (q = 1 -> random intercept;
        q = 2 -> random intercept + slope on Z[:,1]).
    cluster_ids : length-n array identifying subject/cluster.
    """
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    q = Z.shape[1] if Z.ndim > 1 else 1
    theta0 = np.zeros(2 if q == 1 else 4)               # log-sd starting values
    res = optimize.minimize(_reml_neg_ll, theta0,
                             args=(y, X, Z, cluster_ids),
                             method="BFGS", options={"gtol": 1e-6})
    log_theta = res.x
    if q == 1:
        sigma_u2 = math.exp(2 * log_theta[0])
        sigma2 = math.exp(2 * log_theta[1])
        G = np.array([[sigma_u2]])
    else:
        sigma_u2 = math.exp(2 * log_theta[0])
        sigma_v2 = math.exp(2 * log_theta[1])
        rho = math.tanh(log_theta[2])
        sigma2 = math.exp(2 * log_theta[3])
        G = np.array([[sigma_u2, rho * math.sqrt(sigma_u2 * sigma_v2)],
                      [rho * math.sqrt(sigma_u2 * sigma_v2), sigma_v2]])
    # Recompute beta and BLUPs
    unique = np.unique(cluster_ids)
    XtViX = np.zeros((X.shape[1], X.shape[1])); XtViy = np.zeros(X.shape[1])
    for c in unique:
        m = cluster_ids == c
        Z_i = Z[m] if Z.ndim > 1 else Z[m][:, None]
        V_i = Z_i @ G @ Z_i.T + sigma2 * np.eye(m.sum())
        V_inv = np.linalg.inv(V_i)
        XtViX += X[m].T @ V_inv @ X[m]
        XtViy += X[m].T @ V_inv @ y[m]
    beta = np.linalg.solve(XtViX, XtViy)
    cov_beta = np.linalg.inv(XtViX)
    se_beta = np.sqrt(np.clip(np.diag(cov_beta), 0, None))
    # BLUPs per cluster
    blups = {}
    for c in unique:
        m = cluster_ids == c
        Z_i = Z[m] if Z.ndim > 1 else Z[m][:, None]
        V_i = Z_i @ G @ Z_i.T + sigma2 * np.eye(m.sum())
        V_inv = np.linalg.inv(V_i)
        u_hat = G @ Z_i.T @ V_inv @ (y[m] - X[m] @ beta)
        blups[c.item() if hasattr(c, "item") else c] = u_hat.tolist()
    # ICC for random-intercept models
    icc = sigma_u2 / (sigma_u2 + sigma2) if q == 1 else None
    return {"beta": beta.tolist(),
            "SE_beta": se_beta.tolist(),
            "sigma_u2": sigma_u2,
            "sigma2": sigma2,
            "G_matrix": G.tolist(),
            "ICC_random_intercept": icc,
            "REML_log_lik": float(-res.fun),
            "n": int(len(y)), "n_clusters": int(len(unique)),
            "n_params_random": 1 if q == 1 else 3,
            "blups_head": {k: v for k, v in list(blups.items())[:5]},
            "method": "LMM REML via BFGS on profile likelihood"}


def library_versions(y, X, cluster_ids):
    from statsmodels.regression.mixed_linear_model import MixedLM
    import pandas as pd
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y; df["grp"] = cluster_ids
    md = MixedLM(df["y"], df.filter(like="x"), groups=df["grp"])
    fit = md.fit(reml=True)
    return {"statsmodels MixedLM beta": fit.fe_params.tolist(),
            "statsmodels sigma_u2 (variance component)": float(fit.cov_re.iloc[0, 0]),
            "statsmodels sigma2 (residual)": float(fit.scale)}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n_clusters = 40
    n_per = 8
    n = n_clusters * n_per
    cluster_ids = np.repeat(np.arange(n_clusters), n_per)
    u = rng.normal(0, 0.8, n_clusters)                   # random intercepts, sd = 0.8
    x = rng.normal(0, 1, n)
    beta_true = np.array([1.0, 0.5])
    y = beta_true[0] + beta_true[1] * x + u[cluster_ids] + rng.normal(0, 0.4, n)
    X = np.column_stack([np.ones(n), x])
    Z = np.ones((n, 1))                                   # random intercept only

    print("=== LMM (random intercept only) ===")
    fit = fit_lmm(y, X, Z, cluster_ids)
    print(f"  beta = {fit['beta']}  (true = {beta_true.tolist()})")
    print(f"  SE_beta = {fit['SE_beta']}")
    print(f"  sigma_u^2 = {fit['sigma_u2']:.4f}  (true = 0.64)")
    print(f"  sigma^2  = {fit['sigma2']:.4f}  (true = 0.16)")
    print(f"  ICC = {fit['ICC_random_intercept']:.4f}")
    print(f"  REML log-lik = {fit['REML_log_lik']:.3f}")
    print(f"  BLUPs (first 5): {fit['blups_head']}")

    print("\n--- library (statsmodels MixedLM) ---")
    for k, v in library_versions(y, X, cluster_ids).items():
        print(f"  {k}: {v}")

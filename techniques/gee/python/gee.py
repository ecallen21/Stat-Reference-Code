"""Generalized Estimating Equations (Reference §12.8; also covers §12.24, §12.31).

Population-averaged / MARGINAL model for clustered data:

    g(mu_{ij})  =  X_{ij}' beta

Solved by iteratively-reweighted GEE with a WORKING correlation structure R(alpha):

    U(beta) = sum_i X_i' A_i^(1/2) V_i^{-1} A_i^(1/2) (y_i - mu_i(beta)) = 0
        V_i  =  A_i^{1/2} R(alpha) A_i^{1/2}     (working)
        A_i  =  diag(var(mu_i))                   (GLM variance function)

Beauty of GEE (Liang-Zeger 1986): even if R(alpha) is MISSPECIFIED, the beta
estimator is consistent as long as g(mu) is correct. SE via a ROBUST/SANDWICH:

    Var_robust(beta_hat)  =  Bread * Meat * Bread
        Bread = (sum_i X_i' D_i V_i^{-1} D_i X_i)^{-1}
        Meat  = sum_i X_i' D_i V_i^{-1} (y_i - mu_i)(y_i - mu_i)' V_i^{-1} D_i X_i

Also covered:
    §12.24 GEE vs GLMM: GEE gives POPULATION-AVERAGED effects (β = the average
        change in outcome across the population per unit X); GLMM gives
        SUBJECT-SPECIFIC effects (β = the change WITHIN a subject per unit X).
        For linear models the two coincide; for nonlinear links they differ.
    §12.31 When to use: GEE when marginal / policy question; GLMM when
        subject-specific / mechanistic question.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _link_inv_and_var(eta, family: str):
    """Return (mu, dmu/deta, var(mu))."""
    if family == "gaussian":
        return eta, np.ones_like(eta), np.ones_like(eta)
    if family == "binomial":
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        return mu, mu * (1 - mu), mu * (1 - mu)
    if family == "poisson":
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        return mu, mu, mu
    raise ValueError("family must be gaussian / binomial / poisson")


def _working_R(alpha, n_i, structure: str):
    """Return the working correlation matrix for cluster of size n_i."""
    if structure == "independence":
        return np.eye(n_i)
    if structure == "exchangeable":
        R = np.full((n_i, n_i), alpha); np.fill_diagonal(R, 1.0)
        return R
    if structure == "ar1":
        idx = np.arange(n_i)
        return alpha ** np.abs(idx[:, None] - idx[None, :])
    raise ValueError("structure must be independence / exchangeable / ar1")


def gee(y, X, cluster_ids, family: str = "gaussian",
        structure: str = "exchangeable", max_iter: int = 50, tol: float = 1e-6) -> dict:
    """Fit GEE by iteratively-reweighted quasi-likelihood.

    Parameters
    ----------
    y, X : outcome and design (INCLUDE intercept column).
    cluster_ids : cluster identifier per row.
    family : gaussian / binomial / poisson.
    structure : independence / exchangeable / ar1.
    """
    y = np.asarray(y, dtype=float); X = np.asarray(X, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    n, p = X.shape
    unique = np.unique(cluster_ids)

    # Init beta by ordinary GLM (independence GEE == GLM)
    beta = np.linalg.lstsq(X, y if family == "gaussian" else np.log((y + 0.5) / (1.5 - y)) if family == "binomial" else np.log(np.maximum(y, 0.5)), rcond=None)[0]
    alpha = 0.2

    for it in range(max_iter):
        eta = X @ beta
        mu, dmu, var_mu = _link_inv_and_var(eta, family)
        resid_pearson = (y - mu) / np.sqrt(np.clip(var_mu, 1e-12, None))
        # Estimate alpha from cluster residuals (Zeger-Liang 1986)
        if structure == "exchangeable":
            num = 0.0; den = 0.0
            for c in unique:
                m = cluster_ids == c
                r = resid_pearson[m]; n_i = len(r)
                if n_i > 1:
                    num += (np.outer(r, r).sum() - (r * r).sum())
                    den += n_i * (n_i - 1)
            alpha = num / max(den, 1) - 0.0
            alpha = float(np.clip(alpha, -0.999, 0.999))
        elif structure == "ar1":
            num = 0.0; den = 0.0
            for c in unique:
                m = cluster_ids == c
                r = resid_pearson[m]; n_i = len(r)
                if n_i > 1:
                    num += (r[:-1] * r[1:]).sum()
                    den += n_i - 1
            alpha = float(np.clip(num / max(den, 1), -0.999, 0.999))
        else:
            alpha = 0.0

        # Newton step on the GEE score
        HB = np.zeros((p, p))
        score = np.zeros(p)
        for c in unique:
            m = cluster_ids == c
            X_i = X[m]; y_i = y[m]; mu_i = mu[m]; dmu_i = dmu[m]; var_i = var_mu[m]
            n_i = m.sum()
            A_half = np.diag(np.sqrt(np.clip(var_i, 1e-12, None)))
            R = _working_R(alpha, n_i, structure)
            V_i = A_half @ R @ A_half
            V_inv = np.linalg.pinv(V_i)
            D_i = X_i * dmu_i[:, None]                            # d mu / d beta
            HB += D_i.T @ V_inv @ D_i
            score += D_i.T @ V_inv @ (y_i - mu_i)
        try:
            step = np.linalg.solve(HB, score)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(HB) @ score
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new

    # Robust (sandwich) SE
    eta = X @ beta
    mu, dmu, var_mu = _link_inv_and_var(eta, family)
    HB = np.zeros((p, p))
    meat = np.zeros((p, p))
    for c in unique:
        m = cluster_ids == c
        X_i = X[m]; y_i = y[m]; mu_i = mu[m]; dmu_i = dmu[m]; var_i = var_mu[m]
        n_i = m.sum()
        A_half = np.diag(np.sqrt(np.clip(var_i, 1e-12, None)))
        R = _working_R(alpha, n_i, structure)
        V_i = A_half @ R @ A_half
        V_inv = np.linalg.pinv(V_i)
        D_i = X_i * dmu_i[:, None]
        HB += D_i.T @ V_inv @ D_i
        r_i = y_i - mu_i
        u_i = D_i.T @ V_inv @ r_i
        meat += np.outer(u_i, u_i)
    bread = np.linalg.pinv(HB)
    cov_robust = bread @ meat @ bread
    se_robust = np.sqrt(np.clip(np.diag(cov_robust), 0, None))
    z = beta / np.where(se_robust > 0, se_robust, 1e-12)
    p_val = 2 * stats.norm.sf(np.abs(z))
    return {"beta": beta.tolist(),
            "SE_robust": se_robust.tolist(),
            "z": z.tolist(), "p_value": p_val.tolist(),
            "working_alpha": float(alpha),
            "structure": structure, "family": family,
            "n": int(n), "n_clusters": int(len(unique)), "n_iter": it + 1,
            "method": f"GEE ({family}, {structure}) with sandwich SE"}


def library_versions(y, X, cluster_ids, family="binomial"):
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial, Gaussian, Poisson
    import pandas as pd
    fam = {"binomial": Binomial(), "gaussian": Gaussian(), "poisson": Poisson()}[family]
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y; df["grp"] = cluster_ids
    fit = GEE(df["y"], df.filter(like="x"), groups=df["grp"],
               cov_struct=Exchangeable(), family=fam).fit()
    return {"statsmodels GEE beta": fit.params.tolist(),
            "statsmodels GEE SE_robust": fit.bse.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    n_clusters = 50; n_per = 6; n = n_clusters * n_per
    cluster_ids = np.repeat(np.arange(n_clusters), n_per)
    u = rng.normal(0, 0.6, n_clusters)
    x = rng.normal(0, 1, n)
    eta = -0.2 + 0.5 * x + u[cluster_ids]
    p_prob = 1 / (1 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < p_prob).astype(int)
    X = np.column_stack([np.ones(n), x])

    print("=== Binary GEE (exchangeable) ===")
    fit = gee(y, X, cluster_ids, family="binomial", structure="exchangeable")
    print(f"  beta = {fit['beta']}")
    print(f"  SE_robust = {fit['SE_robust']}")
    print(f"  working alpha = {fit['working_alpha']:.4f}")
    print(f"  z = {fit['z']}, p = {fit['p_value']}")

    print("\n--- library (statsmodels GEE) ---")
    for k, v in library_versions(y, X, cluster_ids, "binomial").items():
        print(f"  {k}: {v}")

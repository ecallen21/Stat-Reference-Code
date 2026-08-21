"""Confirmatory Factor Analysis (Reference §19.5).

Contrast with Exploratory Factor Analysis (already implemented): EFA
DISCOVERS latent factor structure from data; CFA TESTS a pre-specified
structure supplied by the analyst.

Model (SEM matrix form)
    x_i = Lambda * eta_i + epsilon_i
    eta_i ~ N(0, Phi)             (factor covariance)
    epsilon_i ~ N(0, Theta)        (error covariance, usually diagonal)

Implied covariance matrix
    Sigma(theta) = Lambda Phi Lambda^T + Theta

ML estimation: minimize F_ML = tr(S Sigma^-1) + log|Sigma| - log|S| - p.

Fit indices
    chi^2 = (n - 1) F_ML                       ~ chi^2(df) if model is correct
    CFI (Comparative Fit Index)                > 0.95 = good
    RMSEA (Root Mean Sq Error of Approximation) < 0.06 = good
    SRMR (Standardized Root Mean Sq Residual)   < 0.08 = good

The demo below implements a small CFA with two correlated factors and
compares to a null (independence) model for CFI.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _fit_ml(S, n, lambda_pattern, phi_free: bool = True):
    """Fit CFA via ML minimizing the ML discrepancy F_ML on the sample covariance S.

    lambda_pattern is a (p x k) 0/1 matrix indicating which loadings are free.
    Factor variances fixed to 1 for identification; factor covariances free.
    Errors are diagonal.
    """
    S = np.asarray(S, dtype=float)
    p, k = lambda_pattern.shape
    n_lambda = int(lambda_pattern.sum())
    n_phi_off = k * (k - 1) // 2
    n_theta = p
    total = n_lambda + n_phi_off + n_theta

    def unpack(theta):
        lam = np.zeros((p, k))
        lam[lambda_pattern.astype(bool)] = theta[:n_lambda]
        Phi = np.eye(k)
        cnt = n_lambda
        for i in range(k):
            for j in range(i + 1, k):
                Phi[i, j] = Phi[j, i] = theta[cnt]; cnt += 1
        Theta = np.diag(theta[cnt:cnt + p] ** 2 + 1e-6)  # square to keep positive
        return lam, Phi, Theta

    def F_ML(theta):
        lam, Phi, Theta = unpack(theta)
        Sigma = lam @ Phi @ lam.T + Theta
        try:
            sign, logdet = np.linalg.slogdet(Sigma)
            if sign <= 0: return 1e10
            Sig_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            return 1e10
        return float(np.trace(S @ Sig_inv) + logdet - np.linalg.slogdet(S)[1] - p)

    theta0 = np.concatenate([np.full(n_lambda, 0.7),
                              np.full(n_phi_off, 0.3),
                              np.sqrt(np.diag(S)) * 0.5])
    res = minimize(F_ML, theta0, method="L-BFGS-B")
    lam, Phi, Theta = unpack(res.x)
    return {"loadings": lam, "factor_cov": Phi, "error_var": np.diag(Theta),
            "F_ML": float(res.fun),
            "chi2": float((n - 1) * res.fun),
            "n_params": int(total)}


def fit_cfa(S, n, lambda_pattern) -> dict:
    """CFA with fit indices."""
    S = np.asarray(S, dtype=float); p = S.shape[0]
    k = lambda_pattern.shape[1]
    fit = _fit_ml(S, n, lambda_pattern)
    # df = p(p+1)/2 - number of estimated parameters
    df = p * (p + 1) // 2 - fit["n_params"]
    chi2 = fit["chi2"]
    # Null (independence) model: only diagonal Sigma
    theta_diag = np.sqrt(np.diag(S))
    Sigma0 = np.diag(theta_diag ** 2)
    F0 = float(np.trace(S @ np.linalg.inv(Sigma0)) + np.linalg.slogdet(Sigma0)[1] -
                np.linalg.slogdet(S)[1] - p)
    chi2_null = (n - 1) * F0; df_null = p * (p + 1) // 2 - p
    # CFI
    cfi = 1 - max(chi2 - df, 0) / max(chi2_null - df_null, 1e-9)
    # RMSEA
    rmsea = math.sqrt(max((chi2 - df) / (df * (n - 1)), 0)) if df > 0 else 0
    # SRMR
    Sigma = fit["loadings"] @ fit["factor_cov"] @ fit["loadings"].T + np.diag(fit["error_var"])
    resid = S - Sigma
    scales = np.sqrt(np.outer(np.diag(S), np.diag(S)))
    srmr = math.sqrt(np.mean((resid[np.triu_indices(p)] / scales[np.triu_indices(p)]) ** 2))
    return {"loadings": fit["loadings"], "factor_cov": fit["factor_cov"],
            "error_var": fit["error_var"],
            "chi2": chi2, "df": int(df),
            "chi2_null": float(chi2_null), "df_null": int(df_null),
            "CFI": float(cfi), "RMSEA": float(rmsea), "SRMR": float(srmr),
            "n_obs": int(n),
            "method": "Confirmatory Factor Analysis (ML)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Simulate 6 items loading on 2 correlated factors (3 each)
    n = 500
    F = rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], n)
    lam = np.array([[0.8, 0], [0.7, 0], [0.75, 0],
                     [0, 0.85], [0, 0.7], [0, 0.65]])
    err_sd = np.array([0.6, 0.7, 0.65, 0.55, 0.7, 0.75])
    X = F @ lam.T + rng.normal(0, 1, (n, 6)) * err_sd
    S = np.cov(X, rowvar=False)

    # Confirmatory pattern matches truth
    pattern = (np.abs(lam) > 0).astype(int)

    r = fit_cfa(S, n=n, lambda_pattern=pattern)
    print("=== CFA with 2 correlated factors, 3 indicators each ===")
    print(f"  chi2 = {r['chi2']:.3f} (df {r['df']})   CFI = {r['CFI']:.4f}   RMSEA = {r['RMSEA']:.4f}   SRMR = {r['SRMR']:.4f}")
    print(f"  factor correlation = {r['factor_cov'][0, 1]:.3f}   (true 0.4)")
    print("  loadings:")
    for i in range(6):
        print(f"    item {i + 1}: {r['loadings'][i].round(3)}")

    print("\n--- library cross-check (lavaan / semopy) ---")
    print("  R: lavaan::cfa(model = '...', data = df)")

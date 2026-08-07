"""Multiple imputation by chained equations (MICE) (Reference §18.6).

Missing-data problem: complete-case analysis discards partial rows and
loses power; single imputation UNDERSTATES uncertainty about the missing
values.

MULTIPLE IMPUTATION (Rubin 1987)
    1. Generate M complete datasets by drawing missing values from their
       PREDICTIVE POSTERIOR given observed data.
    2. Run the analysis on each dataset -> M estimates (theta_m, U_m).
    3. Combine via Rubin's rules:
        theta_bar = mean(theta_m)
        within-imp variance   Wbar = mean(U_m)
        between-imp variance  B    = sample variance of theta_m
        total variance        T    = Wbar + (1 + 1/M) B
        df (Barnard-Rubin)     ~ approx t on (M - 1) large-sample corrected

MICE (Van Buuren 2007) — iterative chained equations
    For each variable with missing values, model it as a function of the
    OTHER variables (predictive mean matching, Bayesian regression, ...);
    cycle through until stability.  Produces one imputed dataset per cycle.

Assumption: MAR (missing at random) given observed variables.  MICE is
misleading under MNAR.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def mice_impute(df, M: int = 5, n_burn: int = 5, seed: int = 0) -> list:
    """MICE via Bayesian linear regression for each continuous column with missing values.

    df : 2-D array with np.nan for missing.
    Returns a list of M imputed 2-D arrays.
    """
    df = np.asarray(df, dtype=float).copy()
    n, p = df.shape
    rng = np.random.default_rng(seed)
    missing = np.isnan(df)
    # Initial imputation: column means
    for j in range(p):
        col_mean = np.nanmean(df[:, j])
        df[missing[:, j], j] = col_mean
    imputations = []
    total_iter = n_burn + M
    for it in range(total_iter):
        for j in range(p):
            if not missing[:, j].any(): continue
            obs = ~missing[:, j]
            X_obs = np.column_stack([np.ones(int(obs.sum())), np.delete(df[obs], j, axis=1)])
            y_obs = df[obs, j]
            # Draw posterior beta and sigma^2 (Normal-Inv-Gamma with diffuse prior)
            beta_hat = np.linalg.solve(X_obs.T @ X_obs + 1e-6 * np.eye(X_obs.shape[1]), X_obs.T @ y_obs)
            resid = y_obs - X_obs @ beta_hat
            n_obs, p_obs = X_obs.shape
            sig2 = float(resid @ resid / max(n_obs - p_obs, 1)) * rng.chisquare(n_obs - p_obs) / (n_obs - p_obs)
            beta_draw = rng.multivariate_normal(beta_hat, sig2 * np.linalg.pinv(X_obs.T @ X_obs) + 1e-8 * np.eye(X_obs.shape[1]))
            # Impute missing values with predictive draw
            X_mis = np.column_stack([np.ones(int(missing[:, j].sum())), np.delete(df[missing[:, j]], j, axis=1)])
            mu = X_mis @ beta_draw
            df[missing[:, j], j] = rng.normal(mu, math.sqrt(sig2))
        if it >= n_burn:
            imputations.append(df.copy())
    return imputations


def rubin_combine(estimates, variances):
    """Rubin's rules to combine M estimates and their within-imp variances."""
    est = np.asarray(estimates); var = np.asarray(variances)
    M = len(est)
    Q_bar = est.mean(axis=0)
    W_bar = var.mean(axis=0)
    B = est.var(axis=0, ddof=1)
    T = W_bar + (1 + 1 / M) * B
    lam = ((1 + 1 / M) * B) / T             # fraction of missing information
    df = (M - 1) * (1 + W_bar / ((1 + 1 / M) * B)) ** 2 if np.all(B > 0) else np.inf
    return {"pooled_estimate": Q_bar,
            "pooled_variance": T,
            "pooled_se": np.sqrt(T),
            "fmi_lambda": lam,
            "df_pool": df,
            "n_imputations": int(M)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(0, 0.5, n)
    y  = 1 + 2 * x1 - 1 * x2 + rng.normal(0, 1, n)
    data = np.column_stack([y, x1, x2])
    # Introduce 30% MAR missingness in x2 (probability depends on x1)
    miss_prob = 1 / (1 + np.exp(-(0.6 * x1)))
    miss_mask = rng.uniform(size=n) < miss_prob
    data_mis = data.copy(); data_mis[miss_mask, 2] = np.nan
    print(f"=== n = {n}, {int(miss_mask.sum())} missing in x2 ===")

    # Complete-case regression (baseline)
    keep = ~np.isnan(data_mis[:, 2])
    X_cc = np.column_stack([np.ones(int(keep.sum())), data_mis[keep, 1], data_mis[keep, 2]])
    beta_cc = np.linalg.solve(X_cc.T @ X_cc, X_cc.T @ data_mis[keep, 0])
    print(f"\n=== Complete-case OLS on {int(keep.sum())} rows ===")
    print(f"  beta = {beta_cc.round(3)}  (true 1, 2, -1)")

    # MICE-based multiple imputation
    print("\n=== MICE + Rubin pooling (M = 10 imputations) ===")
    imps = mice_impute(data_mis, M=10, n_burn=5, seed=1)
    estimates = []; variances = []
    for imp in imps:
        X = np.column_stack([np.ones(n), imp[:, 1], imp[:, 2]])
        beta = np.linalg.solve(X.T @ X, X.T @ imp[:, 0])
        resid = imp[:, 0] - X @ beta
        sig2 = float(resid @ resid / (n - 3))
        var_beta = sig2 * np.diag(np.linalg.pinv(X.T @ X))
        estimates.append(beta); variances.append(var_beta)
    r = rubin_combine(estimates, variances)
    print(f"  pooled beta = {r['pooled_estimate'].round(3)}   (true 1, 2, -1)")
    print(f"  pooled SE   = {r['pooled_se'].round(3)}")
    print(f"  FMI lambda  = {r['fmi_lambda'].round(3)}")

    print("\n--- library cross-check (scikit-learn IterativeImputer / statsmodels MICE) ---")
    try:
        from statsmodels.imputation.mice import MICE, MICEData
        import pandas as pd
        df = pd.DataFrame(data_mis, columns=["y", "x1", "x2"])
        m = MICEData(df)
        for _ in range(5): m.update_all()
        # Just run the imputation loop; skip fit_and_summarize for brevity
        print(f"  statsmodels MICEData: after 5 iters, first imputation x2[0:3] = {m.data['x2'].values[:3].round(3)}")
    except Exception as ex:
        print(f"  (statsmodels MICE unavailable: {ex})")

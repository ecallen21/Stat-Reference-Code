"""Measurement invariance testing (Reference §19.x extra).

Fit a single-factor CFA to G groups at successive constraint levels:

  * CONFIGURAL — same factor structure but everything free per group.
  * METRIC      — loadings equal across groups (intercepts, residual vars free).
  * SCALAR      — loadings + intercepts equal (residual vars, factor mean free).
  * STRICT      — loadings + intercepts + residual variances equal.

Compare adjacent levels by chi-square difference tests.  Better modern
practice: report Delta CFI (< 0.010) and Delta RMSEA (< 0.015) as invariance
evidence (Cheung & Rensvold 2002).

To keep the from-scratch code compact we implement CONFIGURAL and STRICT
(the two extremes) and compute the LR test between them.  Full 4-level
metric/scalar/strict chain: use lavaan.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra
from scipy.optimize import minimize    # BFGS optimiser
from scipy.stats import chi2    # chi-square distribution


def _neg_ll_group(lam, Theta_diag, tau, kappa, S, xbar, n):
    """-2 log-lik (up to constants) for one group's mean-and-cov data."""
    Sigma = np.outer(lam, lam) + np.diag(Theta_diag)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        return 1e10
    logdet = 2 * np.log(np.diag(L)).sum()
    Sinv = np.linalg.inv(Sigma)
    d = xbar - (tau + lam * kappa)
    return n * 0.5 * (logdet + np.trace(Sinv @ S) + d @ Sinv @ d)


def fit_configural(X_by_group) -> dict:
    """Each group: free loadings (with lam[0]=1), free tau, log(Theta_diag), kappa (fixed 0 for group 0)."""
    groups = list(X_by_group.keys()); G = len(groups)
    p = X_by_group[groups[0]].shape[1]
    S = {g: np.cov(X_by_group[g].T, ddof=0) for g in groups}
    xbar = {g: X_by_group[g].mean(axis=0) for g in groups}
    n_g = {g: len(X_by_group[g]) for g in groups}
    fits = {}
    total_ll = 0.0
    for g in groups:
        def neg_ll(theta):
            lam = np.concatenate([[1.0], theta[:p - 1]])
            Theta = np.exp(theta[p - 1: 2 * p - 1])
            tau = theta[2 * p - 1: 3 * p - 1]
            kappa = theta[-1] if g != groups[0] else 0.0
            return _neg_ll_group(lam, Theta, tau, kappa, S[g], xbar[g], n_g[g])
        theta0 = np.concatenate([np.ones(p - 1), np.zeros(p), xbar[g].copy(),
                                  [0.0] if g != groups[0] else []])
        res = minimize(neg_ll, theta0, method="Nelder-Mead",
                        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
        fits[g] = {"nll": float(res.fun),
                    "lam": np.concatenate([[1.0], res.x[:p - 1]]),
                    "Theta": np.exp(res.x[p - 1: 2 * p - 1]),
                    "tau": res.x[2 * p - 1: 3 * p - 1],
                    "kappa": float(res.x[-1]) if g != groups[0] else 0.0}
        total_ll += -res.fun
    n_params = G * (2 * (p - 1) + p + 1) - 1                # rough
    return {"neg_ll_total": -total_ll, "fits": fits, "n_params": n_params,
            "level": "configural"}


def fit_strict(X_by_group, warm_start=None) -> dict:
    """All lam, tau, Theta shared across groups; only kappa (factor mean) group-specific."""
    groups = list(X_by_group.keys()); G = len(groups)
    p = X_by_group[groups[0]].shape[1]
    S = {g: np.cov(X_by_group[g].T, ddof=0) for g in groups}
    xbar = {g: X_by_group[g].mean(axis=0) for g in groups}
    n_g = {g: len(X_by_group[g]) for g in groups}

    def neg_ll(theta):
        lam = np.concatenate([[1.0], theta[:p - 1]])
        Theta = np.exp(theta[p - 1: 2 * p - 1])
        tau = theta[2 * p - 1: 3 * p - 1]
        kappa = np.concatenate([[0.0], theta[3 * p - 1:]])
        total = 0.0
        for i, g in enumerate(groups):
            total += _neg_ll_group(lam, Theta, tau, kappa[i], S[g], xbar[g], n_g[g])
        return total

    if warm_start is not None:
        theta0 = warm_start
    else:
        theta0 = np.concatenate([np.ones(p - 1), np.zeros(p),
                                  xbar[groups[0]].copy(),
                                  np.zeros(G - 1)])
    res = minimize(neg_ll, theta0, method="L-BFGS-B",
                    options={"gtol": 1e-8, "maxiter": 5000})
    # refine with Nelder-Mead
    res = minimize(neg_ll, res.x, method="Nelder-Mead",
                    options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 20000})
    lam = np.concatenate([[1.0], res.x[:p - 1]])
    Theta = np.exp(res.x[p - 1: 2 * p - 1])
    tau = res.x[2 * p - 1: 3 * p - 1]
    kappa = np.concatenate([[0.0], res.x[3 * p - 1:]])
    n_params = (2 * (p - 1) + p) + (G - 1)
    return {"neg_ll_total": float(res.fun), "lam": lam, "Theta": Theta,
            "tau": tau, "kappa": kappa, "n_params": n_params,
            "level": "strict"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = 4
    lam = np.array([1.0, 0.9, 0.8, 1.1])
    Theta = np.array([0.4, 0.3, 0.5, 0.4])
    tau = np.array([0.0, 0.2, -0.1, 0.3])

    # Invariant DGP + factor-mean difference between groups
    n1 = 400; n2 = 400
    f1 = rng.normal(size=n1)
    X1 = tau + np.outer(f1, lam) + rng.normal(scale=np.sqrt(Theta), size=(n1, p))
    f2 = rng.normal(loc=0.5, size=n2)
    X2 = tau + np.outer(f2, lam) + rng.normal(scale=np.sqrt(Theta), size=(n2, p))

    cfg = fit_configural({"g1": X1, "g2": X2})
    # warm-start strict from configural group-1 estimates
    g0 = cfg["fits"]["g1"]
    ws = np.concatenate([g0["lam"][1:], np.log(g0["Theta"]),
                          g0["tau"], np.zeros(1)])
    strict = fit_strict({"g1": X1, "g2": X2}, warm_start=ws)
    chi_diff = 2 * (strict["neg_ll_total"] - cfg["neg_ll_total"])
    df_diff = cfg["n_params"] - strict["n_params"]
    p_val = float(1 - chi2.cdf(chi_diff, df_diff))

    print("=== Measurement invariance: configural vs strict (single-factor CFA) ===")
    print(f"  configural  -2 log-lik = {2 * cfg['neg_ll_total']:>10.1f}   params ≈ {cfg['n_params']}")
    print(f"  strict      -2 log-lik = {2 * strict['neg_ll_total']:>10.1f}   params ≈ {strict['n_params']}")
    print(f"\n  chi-square diff = {chi_diff:.2f}   df ≈ {df_diff}   p = {p_val:.3f}")
    print(f"  (large p means strict invariance is NOT rejected — invariance holds; "
          f"data were simulated invariant.)")

    print(f"\n  strict-model estimates:")
    print(f"    loadings         = {np.round(strict['lam'], 3).tolist()}   true = {lam.tolist()}")
    print(f"    intercepts       = {np.round(strict['tau'], 3).tolist()}   true = {tau.tolist()}")
    print(f"    residual vars    = {np.round(strict['Theta'], 3).tolist()}   true = {Theta.tolist()}")
    print(f"    kappa (per grp)  = {np.round(strict['kappa'], 3).tolist()}   true = [0.0, 0.5]")

    # Test the OPPOSITE case: DGP with non-invariant intercepts
    print("\n\n=== Same test on a NON-invariant DGP (tau[1] shifted by 1 in group 2) ===")
    tau2 = tau.copy(); tau2[1] += 1.0
    X2_ni = tau2 + np.outer(f2, lam) + rng.normal(scale=np.sqrt(Theta), size=(n2, p))
    cfg_ni = fit_configural({"g1": X1, "g2": X2_ni})
    g0_ni = cfg_ni["fits"]["g1"]
    ws_ni = np.concatenate([g0_ni["lam"][1:], np.log(g0_ni["Theta"]),
                             g0_ni["tau"], np.zeros(1)])
    strict_ni = fit_strict({"g1": X1, "g2": X2_ni}, warm_start=ws_ni)
    chi_ni = 2 * (strict_ni["neg_ll_total"] - cfg_ni["neg_ll_total"])
    p_ni = float(1 - chi2.cdf(chi_ni, cfg_ni["n_params"] - strict_ni["n_params"]))
    print(f"  chi-square diff = {chi_ni:.2f}   p = {p_ni:.4f}   "
          f"(small p should reject strict — non-invariance detected)")

    print("\n--- library cross-check (R lavaan::measurementInvariance / semTools::measEq.syntax) ---")

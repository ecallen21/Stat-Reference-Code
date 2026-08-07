"""Panel data: fixed effects, random effects, Hausman test (Reference §12.31, §12.32).

Panel data has repeated observations on the same units (subjects, firms,
countries) over time.  Three standard estimators:

Between (BE): OLS on unit means:  y_bar_i = X_bar_i beta + u_i.
    Uses only cross-unit variation.

Within / Fixed effects (FE): demean within each unit:
    (y_it - y_bar_i) = (X_it - X_bar_i)^T beta + (u_it - u_bar_i)
    Absorbs any TIME-INVARIANT unit-level confounders (b_i).
    Cost: can't estimate coefficients of time-invariant regressors.

Random effects (RE): treat b_i as random with Cov = tau^2 I.  GLS estimator:
    beta_RE = (X^T Omega^-1 X)^-1 X^T Omega^-1 y
    with Omega block-diagonal = sigma^2 I_T + tau^2 J_T.
    Efficient if b_i is UNCORRELATED with X (strong assumption).

Hausman test (1978)
    H_0: RE is consistent (b_i uncorrelated with X).
    H_a: RE is inconsistent, FE is consistent.
    Test statistic: (beta_FE - beta_RE)^T (Cov_FE - Cov_RE)^-1 (beta_FE - beta_RE) ~ chi2(k).
    Reject H_0 -> use FE.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def _demean_within(unit, X, y):
    unit = np.asarray(unit); X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    Xw = X.copy(); yw = y.copy()
    for u in np.unique(unit):
        idx = unit == u
        Xw[idx] -= X[idx].mean(0)
        yw[idx] -= y[idx].mean()
    return Xw, yw


def within_estimator(unit, X, y) -> dict:
    """FE (within) estimator: OLS on demeaned data."""
    Xw, yw = _demean_within(unit, X, y)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    n = len(y); k = X.shape[1]; N = len(np.unique(unit))
    resid = yw - Xw @ beta
    sigma2 = float(resid @ resid / (n - N - k))
    cov = sigma2 * np.linalg.pinv(Xw.T @ Xw)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)),
            "sigma2": sigma2,
            "df": int(n - N - k),
            "method": "Fixed effects (within) estimator"}


def between_estimator(unit, X, y) -> dict:
    """BE: OLS on within-unit means."""
    unit = np.asarray(unit)
    subs = np.unique(unit); N = len(subs)
    X_bar = np.array([X[unit == u].mean(0) for u in subs])
    y_bar = np.array([y[unit == u].mean() for u in subs])
    beta, *_ = np.linalg.lstsq(X_bar, y_bar, rcond=None)
    resid = y_bar - X_bar @ beta
    sigma2 = float(resid @ resid / (N - X_bar.shape[1]))
    cov = sigma2 * np.linalg.pinv(X_bar.T @ X_bar)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)),
            "method": "Between estimator (OLS on unit means)"}


def random_effects(unit, X, y) -> dict:
    """RE estimator: swamy-arora GLS approximation."""
    fe = within_estimator(unit, X, y)
    be = between_estimator(unit, X, y)
    # Estimate variance components (Swamy-Arora style)
    unit = np.asarray(unit); T_bar = float(np.mean([np.sum(unit == u) for u in np.unique(unit)]))
    sigma2_within = fe["sigma2"]
    # sigma^2_total from BE residuals ~ sigma^2_within/T + sigma^2_between
    _, y_bar = _demean_within(unit, X, y)
    tau2 = max(0, np.var(be["beta"] - fe["beta"]))  # crude
    # More standard: tau2 = sigma2_between - sigma2_within / T_bar (Swamy-Arora)
    # Simplified approach: theta = 1 - sqrt(sigma2_within / (sigma2_within + T_bar * tau2))
    tau2 = max(0.0, float(y.var() - sigma2_within))  # very rough
    theta = 1 - math.sqrt(sigma2_within / max(sigma2_within + T_bar * tau2, 1e-8))
    Xg = X.copy(); yg = y.copy()
    for u in np.unique(unit):
        idx = unit == u
        Xg[idx] = X[idx] - theta * X[idx].mean(0)
        yg[idx] = y[idx] - theta * y[idx].mean()
    beta, *_ = np.linalg.lstsq(Xg, yg, rcond=None)
    resid = yg - Xg @ beta
    sigma2 = float(resid @ resid / (len(y) - X.shape[1]))
    cov = sigma2 * np.linalg.pinv(Xg.T @ Xg)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)),
            "theta_transform": float(theta),
            "method": "Random-effects (Swamy-Arora GLS approx)"}


def hausman_test(unit, X, y) -> dict:
    """Hausman FE-vs-RE specification test."""
    fe = within_estimator(unit, X, y)
    re = random_effects(unit, X, y)
    # Approximate covariance of (beta_FE - beta_RE); use Hausman's simplification
    d = fe["beta"] - re["beta"]
    var_d = np.diag(fe["se"] ** 2 - re["se"] ** 2)
    var_d = np.maximum(var_d, 1e-10)
    inv_var = np.diag(1 / np.diag(var_d))
    chi2 = float(d @ inv_var @ d)
    return {"chi2": chi2, "df": len(d),
            "p_value": float(stats.chi2.sf(chi2, df=len(d))),
            "method": "Hausman FE-vs-RE test"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 100; T = 5; n = N * T
    unit = np.repeat(np.arange(N), T)
    # Unit-level unobserved effect correlated with x (endogenous)
    b_i = rng.normal(0, 1, N)
    x = rng.normal(size=n) + 0.5 * b_i[unit]
    z = rng.normal(size=n)
    beta_true = np.array([1.2, -0.5])
    y = beta_true[0] * x + beta_true[1] * z + b_i[unit] + rng.normal(0, 0.5, n)
    X = np.column_stack([x, z])

    print(f"=== Panel data (N = {N} units, T = {T} periods, endogenous b_i correlated with x) ===")
    print("\n=== Pooled OLS (biased) ===")
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"  beta = {beta_ols.round(3)}  (true {beta_true})")

    for name, fn in [("Between", between_estimator), ("Within (FE)", within_estimator),
                      ("Random effects", random_effects)]:
        r = fn(unit, X, y)
        print(f"\n=== {name} ===")
        for j, nm in enumerate(("x", "z")):
            print(f"  {nm}: beta = {r['beta'][j]:.3f} (SE {r['se'][j]:.3f})   true = {beta_true[j]}")

    print("\n=== Hausman test (FE vs RE) ===")
    h = hausman_test(unit, X, y)
    print(f"  chi2 = {h['chi2']:.3f}, df = {h['df']}, p = {h['p_value']:.4f}")
    print(f"  {'REJECT H0 -> use FE' if h['p_value'] < 0.05 else 'do not reject -> RE OK'}")

    print("\n--- library cross-check (linearmodels PanelOLS) ---")
    try:
        import pandas as pd
        from linearmodels.panel import PanelOLS, RandomEffects, BetweenOLS
        df = pd.DataFrame({"unit": unit, "time": np.tile(range(T), N),
                            "x": x, "z": z, "y": y}).set_index(["unit", "time"])
        m_fe = PanelOLS(df.y, df[["x", "z"]], entity_effects=True).fit()
        print(f"  linearmodels FE: {m_fe.params.round(3).tolist()}")
    except Exception as ex:
        print(f"  (linearmodels unavailable: {ex})")

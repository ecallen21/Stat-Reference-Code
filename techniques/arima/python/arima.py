"""ARIMA models (Reference §13.4, §13.5, §13.52).

    AR(p) : x_t = c + phi_1 x_{t-1} + ... + phi_p x_{t-p} + eps_t
    MA(q) : x_t = c + eps_t + theta_1 eps_{t-1} + ... + theta_q eps_{t-q}
    ARMA(p, q) : combination of the two
    ARIMA(p, d, q) : ARMA on the d-th difference of x

Order selection (§13.52):
    Fit models on a grid of (p, d, q), keep the model with smallest AIC (or BIC).
    Rule of thumb: pick d from stationarity tests, then search p, q in [0, 3].

This file:
    - Fits ARMA(p, q) by conditional sum-of-squares MLE (BFGS on -log-lik)
    - Wraps ARIMA(p, d, q) by differencing then fitting ARMA to residuals
    - AIC-based order search
    - Residual Ljung-Box diagnostic
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import optimize, stats    # optimize: BFGS;  stats: distributions


def _arma_neg_ll(params, x, p, q):
    """Conditional-sum-of-squares negative log-lik for ARMA(p, q) with intercept."""
    n = len(x)
    c = params[0]
    phi = params[1:1 + p] if p > 0 else np.array([])
    theta = params[1 + p:1 + p + q] if q > 0 else np.array([])
    log_sigma = params[-1]
    sigma2 = math.exp(2 * log_sigma)
    # Simulate residuals from t = max(p, q) onward
    start = max(p, q)
    if start >= n:
        return 1e10
    eps = np.zeros(n)
    for t in range(start, n):
        pred = c
        for i in range(p): pred += phi[i] * x[t - 1 - i]
        for j in range(q): pred += theta[j] * eps[t - 1 - j]
        eps[t] = x[t] - pred
    resid = eps[start:]
    ll = -0.5 * len(resid) * math.log(2 * math.pi * sigma2) - 0.5 * (resid ** 2).sum() / sigma2
    return -ll


def fit_arma(x, p: int, q: int) -> dict:
    """Fit ARMA(p, q) by conditional-SS MLE via BFGS."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    n_params = 1 + p + q + 1                    # intercept + phi + theta + log_sigma
    theta0 = np.concatenate([
        [x.mean()],                              # intercept
        np.zeros(p),                             # phi
        np.zeros(q),                             # theta
        [math.log(x.std(ddof=1))]                # log sigma
    ])
    res = optimize.minimize(_arma_neg_ll, theta0, args=(x, p, q),
                             method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
    ll = -res.fun
    aic = 2 * n_params - 2 * ll
    bic = math.log(n) * n_params - 2 * ll
    c = res.x[0]
    phi = res.x[1:1 + p].tolist() if p > 0 else []
    theta = res.x[1 + p:1 + p + q].tolist() if q > 0 else []
    sigma = math.exp(res.x[-1])
    return {"c": float(c), "phi": phi, "theta": theta, "sigma": float(sigma),
            "log_lik": float(ll), "AIC": float(aic), "BIC": float(bic),
            "n_params": n_params, "n": int(n),
            "converged": bool(res.success),
            "method": f"ARMA({p}, {q}) via conditional-SS MLE"}


def fit_arima(x, p: int, d: int, q: int) -> dict:
    """ARIMA(p, d, q): difference d times, fit ARMA(p, q) to differenced series."""
    x = np.asarray(x, dtype=float)
    y = x.copy()
    for _ in range(d):
        y = np.diff(y)
    fit = fit_arma(y, p, q)
    fit["d"] = d
    fit["p"] = p
    fit["q"] = q
    fit["method"] = f"ARIMA({p}, {d}, {q}) via differencing + ARMA MLE"
    return fit


def auto_order(x, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> dict:
    """Grid-search (p, d, q) by AIC. Returns the best fit + full table."""
    results = []
    for d in range(max_d + 1):
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0: continue
                try:
                    fit = fit_arima(x, p, d, q)
                    results.append({"p": p, "d": d, "q": q,
                                     "AIC": fit["AIC"], "BIC": fit["BIC"],
                                     "converged": fit["converged"]})
                except Exception:
                    continue
    best = min(results, key=lambda r: r["AIC"])
    return {"grid_results": results, "best_by_AIC": best}


def ljung_box_residuals(fit, x, p, d, q, lags: int = 20) -> dict:
    """Ljung-Box on model residuals."""
    x = np.asarray(x, dtype=float)
    y = x.copy()
    for _ in range(d):
        y = np.diff(y)
    # recompute residuals with fitted params
    c = fit["c"]; phi = fit["phi"]; theta = fit["theta"]
    start = max(p, q)
    eps = np.zeros(len(y))
    for t in range(start, len(y)):
        pred = c
        for i in range(p): pred += phi[i] * y[t - 1 - i]
        for j in range(q): pred += theta[j] * eps[t - 1 - j]
        eps[t] = y[t] - pred
    resid = eps[start:]
    n = len(resid)
    xbar = resid.mean()
    rhos = np.array([
        ((resid[k:] - xbar) * (resid[:n - k] - xbar)).sum() / ((resid - xbar) ** 2).sum()
        for k in range(lags + 1)
    ])
    Q = n * (n + 2) * sum(rhos[k] ** 2 / (n - k) for k in range(1, lags + 1))
    return {"Q": float(Q), "df": lags - p - q,
             "p_value": float(stats.chi2.sf(Q, max(1, lags - p - q))),
             "interpretation": "large p => residuals look white (good)"}


def library_versions(x, order=(1, 0, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    m = ARIMA(x, order=order).fit()
    return {"statsmodels ARIMA params": m.params.tolist(),
            "statsmodels AIC": float(m.aic),
            "statsmodels sigma^2": float(m.mse)}


if __name__ == "__main__":
    rng = np.random.default_rng(13)
    n = 300
    # ARMA(1, 1): x_t = 0.6 x_{t-1} + eps_t + 0.4 eps_{t-1}
    eps = rng.normal(0, 1, n); x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + eps[t] + 0.4 * eps[t - 1]

    print("=== ARMA(1, 1) fit (true phi = 0.6, theta = 0.4) ===")
    fit = fit_arma(x, p=1, q=1)
    print(f"  c = {fit['c']:+.4f}")
    print(f"  phi = {fit['phi']}")
    print(f"  theta = {fit['theta']}")
    print(f"  sigma = {fit['sigma']:.4f}")
    print(f"  AIC = {fit['AIC']:.2f}")

    print("\n=== AIC order search up to (3, 2, 3) ===")
    ao = auto_order(x, max_p=3, max_d=1, max_q=3)
    print(f"  best (p, d, q) by AIC: ({ao['best_by_AIC']['p']}, "
          f"{ao['best_by_AIC']['d']}, {ao['best_by_AIC']['q']})   "
          f"AIC = {ao['best_by_AIC']['AIC']:.2f}")

    print("\n=== Ljung-Box on residuals ===")
    lb = ljung_box_residuals(fit, x, 1, 0, 1)
    print(f"  Q = {lb['Q']:.2f}, df = {lb['df']}, p = {lb['p_value']:.4f}")
    print(f"  {lb['interpretation']}")

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(x, order=(1, 0, 1)).items():
        print(f"  {k}: {v}")

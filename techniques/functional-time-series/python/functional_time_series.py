"""Functional time series forecasting via FPCA + AR on the scores
(Hyndman-Ullah 2007; Reference §13.x extra).

A functional TS is a sequence of curves X_t(u), u in [a, b], t = 1, ..., T
(e.g. daily electricity-demand load profile for u = time of day; mortality
rate by age; interest-rate yield curve by maturity).

Model:
    X_t(u) = mu(u) + sum_{k=1}^K xi_{t,k} * phi_k(u) + eps_t(u)

  * mu(u): mean function
  * phi_k(u): FPCA loadings (top-K eigenfunctions of the empirical covariance)
  * xi_{t,k}: FPCA scores (K time series)

Forecast:
    xi_{t+h,k} ~ AR(p) fit  ->  xi_hat_{T+h,k}
    X_hat_{T+h}(u) = mu(u) + sum_k xi_hat_{T+h,k} * phi_k(u)

Discrete implementation: represent each curve as a length-D vector on a fixed
grid; FPCA reduces to eigendecomposition of the covariance matrix.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def fpca(X, K: int) -> dict:
    """X: T x D matrix (T curves, D-length discretisation)."""
    X = np.asarray(X, dtype=float); T, D = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    C = Xc.T @ Xc / (T - 1)                             # (D, D) covariance
    w, V = np.linalg.eigh(C)                             # ascending eigenvalues
    order = np.argsort(-w)
    w = w[order]; V = V[:, order]
    phi = V[:, :K]
    scores = Xc @ phi                                    # (T, K)
    var_expl = w[:K] / w.sum()
    return {"mu": mu, "phi": phi, "scores": scores,
            "eigvals": w[:K], "var_explained": var_expl}


def _ar_fit(y, p: int = 2):
    """Fit AR(p) by OLS on lagged design; return coefs and residual variance."""
    y = np.asarray(y, dtype=float); T = len(y)
    Y = y[p:]
    X = np.column_stack([y[p - k - 1: T - k - 1] for k in range(p)])
    X = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return beta, float(resid.var(ddof=1))


def _ar_forecast(y, beta, p: int, h: int):
    y = list(np.asarray(y, dtype=float))
    for _ in range(h):
        past = [y[-k - 1] for k in range(p)]
        y.append(float(beta[0] + sum(beta[k + 1] * past[k] for k in range(p))))
    return y[-h:]


def fts_forecast(X, K: int = 3, ar_order: int = 2, h: int = 5) -> dict:
    fp = fpca(X, K)
    forecasts_by_k = []
    for k in range(K):
        beta, _ = _ar_fit(fp["scores"][:, k], p=ar_order)
        forecasts_by_k.append(_ar_forecast(fp["scores"][:, k], beta, ar_order, h))
    forecasts_by_k = np.array(forecasts_by_k)             # (K, h)
    # reconstruct curves
    X_forecast = fp["mu"][None, :] + forecasts_by_k.T @ fp["phi"].T
    return {"fpca": fp, "forecasts_scores": forecasts_by_k,
            "X_forecast": X_forecast, "h": h,
            "method": "FTS forecast via FPCA + AR on scores"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 80; D = 24
    u = np.linspace(0, 1, D)
    mu = 5 + 3 * np.sin(2 * np.pi * u)                   # baseline daily profile
    phi1 = np.cos(2 * np.pi * u); phi2 = np.sin(4 * np.pi * u)
    # AR(1) evolution of two score processes
    xi1 = np.zeros(T); xi2 = np.zeros(T)
    for t in range(1, T):
        xi1[t] = 0.9 * xi1[t - 1] + rng.normal(scale=0.5)
        xi2[t] = 0.7 * xi2[t - 1] + rng.normal(scale=0.3)
    X = mu[None, :] + np.outer(xi1, phi1) + np.outer(xi2, phi2) \
        + rng.normal(scale=0.15, size=(T, D))

    # rolling h=1 forecast: at each time t, train on X[:t], forecast X[t]
    T_train_start = 40; h = 1
    rmse_fts = []; rmse_base = []
    for t in range(T_train_start, T):
        fit_t = fts_forecast(X[:t], K=3, ar_order=2, h=h)
        rmse_fts.append(np.sqrt(((X[t: t + h] - fit_t["X_forecast"]) ** 2).mean()))
        base = np.tile(fit_t["fpca"]["mu"], (h, 1))
        rmse_base.append(np.sqrt(((X[t: t + h] - base) ** 2).mean()))
    print(f"=== Functional TS: rolling one-step forecasts (windows {T_train_start}..{T - 1}) ===")
    print(f"  D={D}, K=3, AR order 2")
    print(f"  variance explained by top 3 FPCs (final window): "
          f"{np.round(fit_t['fpca']['var_explained'], 4).tolist()}")
    print(f"  RMSE (FPCA + AR one-step): {np.mean(rmse_fts):.4f}")
    print(f"  RMSE (mean-only baseline): {np.mean(rmse_base):.4f}")
    print(f"  relative reduction: {(1 - np.mean(rmse_fts) / np.mean(rmse_base)) * 100:.1f}%")

    print("\n--- library cross-check (R ftsa::forecast.ftsm; Python skfda) ---")

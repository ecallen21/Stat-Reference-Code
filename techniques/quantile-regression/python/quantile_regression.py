"""Quantile regression (Reference §5.15; Koenker & Bassett 1978).

OLS estimates the CONDITIONAL MEAN E[y | X].  Quantile regression estimates
the CONDITIONAL QUANTILE Q_tau(y | X) at a chosen tau in (0, 1):

    minimize sum_i rho_tau(y_i - X_i beta)
        rho_tau(u) = u (tau - I(u < 0))
                   = max(tau u, (tau - 1) u)     (pinball / check loss)

    tau = 0.5 -> conditional median  (L1 regression, robust to outliers)
    tau = 0.1 or 0.9 -> tails, useful for heterogeneous effects

The problem is a LINEAR PROGRAM (Koenker-Bassett).  Solve via scipy.optimize
linprog, or via smoothed pinball loss + BFGS (used here for simplicity).

Reporting a whole GRID of tau (0.1, 0.25, 0.5, 0.75, 0.9) reveals how the
covariate effect changes across the outcome distribution -- a scale-effect
looks like slopes fanning out with tau.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _pinball(u, tau):
    return np.maximum(tau * u, (tau - 1) * u)


def quantile_regression(X, y, tau: float = 0.5, method: str = "smoothed",
                        h: float = 0.05) -> dict:
    """Quantile regression via smoothed pinball loss (Muggeo 2017)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    if method == "smoothed":
        # Nesterov's smoothed pinball loss for BFGS
        def loss(beta):
            u = y - X @ beta
            # Huber-smoothed: rho_tau,h(u) = tau u + h log(1 + exp(-u/h)) (for tau=0.5 gives Huber)
            # Use: (tau - I(u<0))(u) + h * SmoothCorrection
            positive = u >= 0
            L = np.where(positive, tau * u + h * np.log1p(np.exp(-u / h)),
                         (tau - 1) * u + h * np.log1p(np.exp(u / h)))
            return L.sum()
        beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = minimize(loss, beta0, method="BFGS")
        beta = res.x
    elif method == "lp":
        from scipy.optimize import linprog
        # Formulate as LP: min tau sum u+ + (1-tau) sum u- s.t. y = X beta + u+ - u-
        c = np.concatenate([np.zeros(p), tau * np.ones(n), (1 - tau) * np.ones(n)])
        A_eq = np.column_stack([X, np.eye(n), -np.eye(n)])
        bounds = [(None, None)] * p + [(0, None)] * (2 * n)
        res = linprog(c, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
        beta = res.x[:p]
    else: raise ValueError("method must be 'smoothed' or 'lp'")
    resid = y - X @ beta
    return {"beta": beta, "tau": float(tau),
            "check_loss": float(_pinball(resid, tau).sum()),
            "method": f"Quantile regression ({method})"}


def quantile_regression_grid(X, y, taus=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list:
    """Fit QR at each tau and return per-tau row."""
    rows = []
    for tau in taus:
        r = quantile_regression(X, y, tau=tau, method="smoothed")
        rows.append({"tau": tau, "beta": r["beta"], "check_loss": r["check_loss"]})
    return rows


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n); X = np.column_stack([np.ones(n), x])
    # Heteroscedastic errors: variance grows with x -> slope should fan out across tau
    y = 1 + 2 * x + (1 + 0.8 * x) * rng.normal(size=n)

    print("=== Quantile regression at tau = 0.5 (median) vs OLS ===")
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = quantile_regression(X, y, tau=0.5, method="smoothed")
    print(f"  OLS beta       : {beta_ols.round(3)}")
    print(f"  QR tau=0.5 beta: {r['beta'].round(3)}")

    print("\n=== Slope profile across quantiles (fanning under heteroscedasticity) ===")
    for row in quantile_regression_grid(X, y):
        print(f"  tau = {row['tau']:.2f}  intercept = {row['beta'][0]:6.3f}  slope = {row['beta'][1]:6.3f}")

    print("\n--- library cross-check (statsmodels QuantReg) ---")
    try:
        import statsmodels.api as sm
        for t in (0.1, 0.5, 0.9):
            m = sm.QuantReg(y, X).fit(q=t)
            print(f"  statsmodels tau={t}: {m.params.round(3)}")
    except Exception as ex:
        print(f"  (statsmodels QuantReg unavailable: {ex})")

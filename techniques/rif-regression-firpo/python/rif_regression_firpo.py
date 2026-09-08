"""Unconditional (RIF) quantile regression (Reference Sec 47.46).

Firpo, Fortin & Lemieux 2009 'Unconditional quantile regressions',
Econometrica 77(3). Contrasts *conditional* quantile regression
(Koenker-Bassett) which models Q_Y|X(tau|x), with the UNCONDITIONAL
quantile of Y. Uses the RECENTERED INFLUENCE FUNCTION (RIF):

    RIF(y; q_tau, F_Y) = q_tau + (tau - 1{y <= q_tau}) / f_Y(q_tau)

Running OLS with the RIF as outcome gives partial derivatives of
the marginal quantile with respect to each covariate -- an
'unconditional partial effect'.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.stats import gaussian_kde    # density at q_tau


def rif_ols(X, y, tau):
    """RIF-OLS unconditional quantile regression at level tau."""
    q = np.quantile(y, tau)
    f = gaussian_kde(y).evaluate([q])[0]
    rif = q + (tau - (y <= q).astype(float)) / f
    Xc = np.column_stack([np.ones_like(y), X])
    beta, *_ = np.linalg.lstsq(Xc, rif, rcond=None)
    resid = rif - Xc @ beta
    return {"beta": beta, "q_tau": float(q), "f_at_q": float(f),
            "sigma2": float((resid ** 2).mean())}


def conditional_quantile_slope(X, y, tau):
    """Standard Koenker-Bassett conditional quantile regression via LP surrogate."""
    from scipy.optimize import minimize
    def loss(b):
        r = y - X @ b[1:] - b[0]
        return np.sum(np.where(r >= 0, tau * r, (tau - 1) * r))
    res = minimize(loss, np.zeros(X.shape[1] + 1), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 20000})
    return res.x


if __name__ == "__main__":
    print("=== RIF unconditional quantile regression (Firpo-Fortin-Lemieux 2009) ===\n")
    rng = np.random.default_rng(0)
    n = 4000
    x1 = rng.normal(size=n)                    # continuous covariate
    x2 = rng.binomial(1, 0.5, size=n)          # binary (e.g. treatment)
    y = 2.0 + 1.0 * x1 + 0.5 * x2 + (0.5 + 0.3 * x2) * rng.normal(size=n)
    X = np.column_stack([x1, x2])

    for tau in [0.10, 0.50, 0.90]:
        rif = rif_ols(X, y, tau)
        cq = conditional_quantile_slope(X, y, tau)
        print(f"  tau={tau:.2f}   q_tau(Y) = {rif['q_tau']:+.3f}")
        print(f"    RIF-OLS beta      = intercept {rif['beta'][0]:+.3f}, "
              f"x1 {rif['beta'][1]:+.3f}, x2 {rif['beta'][2]:+.3f}")
        print(f"    Cond quantile beta = intercept {cq[0]:+.3f}, "
              f"x1 {cq[1]:+.3f}, x2 {cq[2]:+.3f}")

    print("\n  RIF slopes give the change in the marginal quantile of Y")
    print("  per unit change in x. Differs from conditional quantile slopes")
    print("  under heteroskedasticity or shape shifts.")
    print("\n--- library cross-check (dineq / uqr R; statsmodels + custom Python) ---")

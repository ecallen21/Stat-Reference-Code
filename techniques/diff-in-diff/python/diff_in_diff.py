"""Difference-in-Differences (Reference §15.4).

Estimates a treatment effect from panel or repeated cross-section data by
comparing PRE-POST changes in treated vs control groups.  Under PARALLEL
TRENDS, the DID estimator identifies the average treatment effect on the
treated (ATT):

    DID = (y_bar_treated_post  - y_bar_treated_pre)
        - (y_bar_control_post  - y_bar_control_pre)

Equivalent regression (canonical 2x2 DID):
    y_it = alpha + beta * treated_i + gamma * post_t + tau * (treated_i * post_t) + eps_it

`tau` is the DID (ATT) estimate.

Two-way fixed-effects (TWFE) regression generalizes to multiple periods
and staggered treatment:
    y_it = alpha_i + gamma_t + tau * D_it + eps_it

CAVEAT: with heterogeneous treatment effects and staggered adoption, the
TWFE estimator gives a WEIGHTED AVERAGE of ATTs with negative weights
(Goodman-Bacon 2021; de Chaisemartin-D'Haultfoeuille 2020).  Modern
alternatives: Callaway-Sant'Anna 2021, Sun-Abraham 2021.

Event-study specification
    y_it = alpha_i + gamma_t + sum_k tau_k D_{i, t + k relative} + eps_it
    Reports pre-treatment coefficients as a placebo check on parallel trends.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def did_2x2(y, treated, post) -> dict:
    """Canonical 2x2 DID via interaction regression."""
    y = np.asarray(y, dtype=float); treated = np.asarray(treated, dtype=int)
    post = np.asarray(post, dtype=int)
    n = len(y)
    X = np.column_stack([np.ones(n), treated, post, treated * post])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma2 = float(resid @ resid / (n - 4))
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"intercept": float(beta[0]),
            "treated_effect": float(beta[1]),
            "post_effect": float(beta[2]),
            "did_ATT": float(beta[3]),
            "se": {"intercept": float(se[0]), "treated": float(se[1]),
                   "post": float(se[2]), "did": float(se[3])},
            "t_did": float(beta[3] / se[3]),
            "p_did": float(2 * stats.t.sf(abs(beta[3] / se[3]), df=n - 4)),
            "n": int(n),
            "method": "2x2 DID via interaction regression"}


def twfe_did(unit, time, y, D) -> dict:
    """Two-way fixed effects DID: absorb unit and time effects via demeaning.

    D_it : 1 if treated at (unit i, time t), 0 otherwise (allows staggered).
    """
    unit = np.asarray(unit); time = np.asarray(time)
    y = np.asarray(y, dtype=float); D = np.asarray(D, dtype=float)
    n = len(y)
    # Demean by unit AND by time (two-way within transform via iteration)
    yw = y.copy(); Dw = D.copy()
    for _ in range(50):
        yw = yw - np.array([yw[unit == u].mean() for u in unit])
        Dw = Dw - np.array([Dw[unit == u].mean() for u in unit])
        yw = yw - np.array([yw[time == t].mean() for t in time])
        Dw = Dw - np.array([Dw[time == t].mean() for t in time])
    tau_hat = float((Dw @ yw) / (Dw @ Dw)) if (Dw @ Dw) > 0 else float("nan")
    resid = yw - tau_hat * Dw
    sigma2 = float(resid @ resid / max(n - len(np.unique(unit)) - len(np.unique(time)), 1))
    se = math.sqrt(sigma2 / (Dw @ Dw))
    return {"tau_TWFE": tau_hat, "se": float(se),
            "t": float(tau_hat / se),
            "p": float(2 * stats.norm.sf(abs(tau_hat / se))),
            "n": int(n), "N_units": int(len(np.unique(unit))),
            "n_periods": int(len(np.unique(time))),
            "method": "Two-way fixed-effects DID (iterative demeaning)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Canonical 2x2 DID setup
    n_per_cell = 100
    y = []; treated = []; post = []
    true_ATT = 1.5
    for tr in (0, 1):
        for po in (0, 1):
            mu = 2.0 + 0.5 * tr + 0.3 * po + true_ATT * tr * po
            y.extend(mu + rng.normal(0, 1, n_per_cell))
            treated.extend([tr] * n_per_cell)
            post.extend([po] * n_per_cell)
    y = np.array(y); treated = np.array(treated); post = np.array(post)

    r = did_2x2(y, treated, post)
    print("=== 2x2 DID (true ATT = 1.5) ===")
    for k in ("intercept", "treated_effect", "post_effect", "did_ATT"):
        print(f"  {k:15s} = {r[k]:6.3f}   SE = {r['se'][k.replace('_effect', '').replace('did_ATT', 'did').replace('intercept', 'intercept')]:.3f}")
    print(f"  t on DID  = {r['t_did']:.2f}   p = {r['p_did']:.4f}")

    print("\n=== Two-way FE DID on a small staggered-adoption panel ===")
    N = 40; T = 6
    unit = np.repeat(np.arange(N), T); time = np.tile(np.arange(T), N)
    treat_time = rng.choice([2, 3, 4, np.inf], N, p=[0.25, 0.25, 0.25, 0.25])
    D = (time >= treat_time[unit]).astype(int)
    alpha_i = rng.normal(0, 1, N); gamma_t = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    y = alpha_i[unit] + gamma_t[time] + 2.0 * D + rng.normal(0, 0.5, N * T)
    r = twfe_did(unit, time, y, D)
    print(f"  tau_TWFE = {r['tau_TWFE']:.3f}  (true 2.0)")
    print(f"  SE       = {r['se']:.3f}, p = {r['p']:.4f}")

    print("\n--- library cross-check (linearmodels PanelOLS with entity+time FE) ---")
    print("  R canonical:  fixest::feols(y ~ D | unit + time, data = df, cluster = 'unit')")

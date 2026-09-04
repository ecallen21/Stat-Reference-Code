"""Hausman test (Reference Sec 35.2).

Hausman (1978) 'Specification tests in econometrics.'

Test whether the RANDOM-EFFECTS (RE) estimator is CONSISTENT vs the
FIXED-EFFECTS (FE) estimator (which is always consistent under
strict exogeneity):

  H_0: E[a_i | x_it] = 0     -> RE consistent + efficient
  H_1: E[a_i | x_it] != 0    -> RE inconsistent; use FE

Statistic:
  H = (beta_FE - beta_RE)' * (V_FE - V_RE)^-1 * (beta_FE - beta_RE)
  H ~ chi^2(k) under H_0     (k = # regressors)

Here we implement a compact panel-data FE / RE estimator and Hausman
test on synthetic data: (a) unit effects uncorrelated with x -> fail
to reject; (b) correlated -> reject.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import chi2 as _chi2


def within_transform(X, y, unit):
    """Demean each variable within units (fixed-effects transform)."""
    unique = np.unique(unit)
    Xw = X.copy().astype(float)
    yw = y.copy().astype(float)
    for u in unique:
        m = unit == u
        Xw[m] = Xw[m] - Xw[m].mean(axis=0)
        yw[m] = yw[m] - yw[m].mean()
    return Xw, yw


def fit_fe(X, y, unit):
    Xw, yw = within_transform(X, y, unit)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = yw - Xw @ beta
    n = X.shape[0]; k = X.shape[1]
    n_units = len(np.unique(unit))
    sigma2 = float(resid @ resid / (n - k - n_units))
    V = sigma2 * np.linalg.inv(Xw.T @ Xw + 1e-8 * np.eye(k))
    return beta, V


def fit_re(X, y, unit):
    """Random-effects GLS via simple sigma^2_u / sigma^2_e estimator + partial demean."""
    n, k = X.shape
    Xd, yd = within_transform(X, y, unit)
    beta_w, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    resid_w = yd - Xd @ beta_w
    sigma2_e = float(resid_w @ resid_w / (n - k))
    # Between: unit means regression
    ub = np.array([[X[unit == u].mean(axis=0), y[unit == u].mean()] for u in np.unique(unit)], dtype=object)
    Xb = np.stack([r[0] for r in ub])
    yb = np.array([r[1] for r in ub], dtype=float)
    beta_b, *_ = np.linalg.lstsq(Xb, yb, rcond=None)
    resid_b = yb - Xb @ beta_b
    T = float(len(y) / len(np.unique(unit)))
    sigma2_between = float(resid_b @ resid_b / max(len(yb) - k, 1))
    sigma2_u = max(0.0, sigma2_between - sigma2_e / T)
    theta = 1 - np.sqrt(sigma2_e / (sigma2_e + T * sigma2_u))
    # Partial demean by theta
    Xr, yr = X.copy().astype(float), y.copy().astype(float)
    for u in np.unique(unit):
        m = unit == u
        Xr[m] = Xr[m] - theta * Xr[m].mean(axis=0)
        yr[m] = yr[m] - theta * yr[m].mean()
    beta, *_ = np.linalg.lstsq(Xr, yr, rcond=None)
    resid = yr - Xr @ beta
    sigma2 = float(resid @ resid / (n - k))
    V = sigma2 * np.linalg.inv(Xr.T @ Xr + 1e-8 * np.eye(k))
    return beta, V


def hausman_test(X, y, unit):
    """Wooldridge auxiliary-regression Hausman: augment RE-transformed regression
    with the within-demeaned regressors; joint significance of those coefficients
    is a robust Hausman statistic (Wooldridge 2002, Baltagi 2005)."""
    beta_FE, _ = fit_fe(X, y, unit)
    beta_RE, _ = fit_re(X, y, unit)
    # Build augmented regression: y ~ X (RE) + X_within (auxiliary)
    Xw, _ = within_transform(X, y, unit)
    Xaug = np.hstack([X, Xw])
    beta, *_ = np.linalg.lstsq(Xaug, y, rcond=None)
    resid = y - Xaug @ beta
    n, m = Xaug.shape
    sigma2 = float(resid @ resid / (n - m))
    V = sigma2 * np.linalg.inv(Xaug.T @ Xaug + 1e-8 * np.eye(m))
    k = X.shape[1]
    gamma = beta[k:]                                # auxiliary coefficients
    Vg = V[k:, k:]
    H = float(gamma @ np.linalg.solve(Vg, gamma))
    p = float(1 - _chi2.cdf(H, k))
    return {"H": H, "df": k, "p": p, "beta_FE": beta_FE, "beta_RE": beta_RE}


def _make_panel(rng, n_units, T, corr_alpha):
    """corr_alpha in {0, 1}: 0 = unit effects independent of x; 1 = correlated."""
    alpha = rng.normal(0, 1, n_units)
    x_list = []; y_list = []; u_list = []
    for i in range(n_units):
        x = rng.normal(0, 1, T) + corr_alpha * alpha[i]
        y = 1.5 * x + alpha[i] + rng.normal(0, 0.5, T)
        x_list.append(x); y_list.append(y); u_list.append(np.full(T, i))
    X = np.stack([np.concatenate(x_list)], axis=1)
    y = np.concatenate(y_list)
    unit = np.concatenate(u_list)
    return X, y, unit


if __name__ == "__main__":
    print("=== Hausman test (Hausman 1978) ===\n")
    rng = np.random.default_rng(0)
    for name, corr in (("Case A: unit effects INDEPENDENT of x", 0.0),
                        ("Case B: unit effects CORRELATED with x", 1.0)):
        X, y, unit = _make_panel(rng, n_units=30, T=8, corr_alpha=corr)
        r = hausman_test(X, y, unit)
        print(f"  {name}")
        print(f"    beta_FE = {r['beta_FE'][0]:.3f}   beta_RE = {r['beta_RE'][0]:.3f}"
              f"   H = {r['H']:.3f}   df = {r['df']}   p = {r['p']:.3f}")
        print(f"    -> {'REJECT RE (use FE)' if r['p'] < 0.05 else 'fail to reject; RE OK'}\n")

    print("--- library cross-check (R plm::phtest; Python linearmodels.iv.PanelOLS + Hausman helper) ---")

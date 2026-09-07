"""Indirect inference (Reference Sec 45.8).

Gouriéroux, Monfort & Renault 1993; Smith 1993. When the structural
model's likelihood is intractable, use an AUXILIARY tractable model as
an 'instrument' for parameter estimation. Steps:

    1. Fit the auxiliary model to the DATA -> beta_hat.
    2. Simulate S data sets from the structural model at candidate
       theta and fit the auxiliary model to each -> {beta_hat_s(theta)}.
    3. Choose theta to make simulated auxiliary parameters close to
       the data auxiliary parameters:

         theta_hat = argmin (beta_hat - beta_bar(theta))' W (...)

Related to Method of Simulated Moments; the auxiliary model provides
the moments implicitly. Consistent under Gouriéroux et al. regularity;
efficiency approaches MLE as the auxiliary model becomes an
'encompassing' model.

We demonstrate on a MOVING-AVERAGE(1) process y_t = eps_t + theta *
eps_{t-1}. The likelihood is available but slightly cumbersome; use
an AR(2) as the AUXILIARY model and recover theta by matching AR(2)
coefficients.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize_scalar    # 1-D optimisation


def fit_ar2(y):
    """OLS AR(2) coefficients: y_t = phi_1 y_{t-1} + phi_2 y_{t-2} + noise."""
    y = np.asarray(y)
    X = np.c_[y[1:-1], y[:-2]]
    yt = y[2:]
    b, *_ = np.linalg.lstsq(X, yt, rcond=None)
    return b     # (phi_1, phi_2)


def simulate_ma1(theta, T, rng):
    eps = rng.normal(size=T + 1)
    y = eps[1:] + theta * eps[:-1]
    return y


def indirect_inference(y, S=40, seed=0):
    beta_hat = fit_ar2(y)
    T = len(y)

    def criterion(theta):
        rng = np.random.default_rng(seed)         # common random numbers
        diffs = np.zeros(2)
        for s in range(S):
            y_s = simulate_ma1(theta, T, rng)
            beta_s = fit_ar2(y_s)
            diffs += (beta_hat - beta_s)
        d = diffs / S
        return float(d @ d)

    r = minimize_scalar(criterion, bounds=(-0.95, 0.95), method="bounded")
    return {"theta_hat": r.x, "criterion": r.fun, "beta_data": beta_hat}


if __name__ == "__main__":
    print("=== Indirect inference -- MA(1) via AR(2) auxiliary ===\n")
    rng = np.random.default_rng(0)
    theta_true = 0.6
    T = 800
    y = simulate_ma1(theta_true, T, rng)

    r = indirect_inference(y, S=40, seed=42)
    print(f"  True theta      = {theta_true:.3f}")
    print(f"  II  theta_hat   = {r['theta_hat']:.3f}")
    print(f"  Auxiliary AR(2) on data: (phi_1, phi_2) = "
          f"({r['beta_data'][0]:+.3f}, {r['beta_data'][1]:+.3f})")
    print(f"\n  MA(1) theory: phi_1 = theta / (1 + theta^2) = "
          f"{theta_true / (1 + theta_true ** 2):.3f}")

    print("\n--- library cross-check (indirectInference R; sindyr / from-scratch Python) ---")

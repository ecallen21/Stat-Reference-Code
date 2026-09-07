"""G-computation / parametric G-formula (Reference Sec 15.20).

Robins 1986.  Standardise the outcome model's predictions to a
hypothetical population under each treatment level:

    ATE = E[ m(X, 1) - m(X, 0) ]

where m(X, T) = E[Y | X, T] is fitted on the observed data.
Contrast to IPTW: g-computation uses OUTCOME modelling; IPTW uses
TREATMENT modelling.

For time-varying treatments and confounders, extend to the
sequential g-formula.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LinearRegression


def g_computation(X, T, y):
    Xt = np.column_stack([X, T])
    m = LinearRegression().fit(Xt, y)
    m1 = m.predict(np.column_stack([X, np.ones(len(y))]))
    m0 = m.predict(np.column_stack([X, np.zeros(len(y))]))
    return {"ATE": float(np.mean(m1 - m0)),
            "counterfactual_mean_1": float(m1.mean()),
            "counterfactual_mean_0": float(m0.mean())}


def g_computation_bootstrap(X, T, y, B=200, seed=0):
    rng = np.random.default_rng(seed)
    ates = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ates.append(g_computation(X[idx], T[idx], y[idx])["ATE"])
    ates = np.array(ates)
    return {"ATE": float(np.mean(ates)),
            "SE": float(np.std(ates)),
            "CI95": (float(np.quantile(ates, 0.025)), float(np.quantile(ates, 0.975)))}


if __name__ == "__main__":
    print("=== G-computation / parametric G-formula ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(0, 1, (n, 3))
    logit_T = 0.5 * X[:, 0] + 0.4 * X[:, 1]
    T = (rng.random(n) < 1 / (1 + np.exp(-logit_T))).astype(int)
    # True ATE = 0.5
    y = 1.0 + 0.5 * T + 0.6 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 1, n)

    r_point = g_computation(X, T, y)
    r_boot = g_computation_bootstrap(X, T, y, B=200)
    print(f"  Point ATE estimate = {r_point['ATE']:+.3f}")
    print(f"    counterfactual mean under T=1: {r_point['counterfactual_mean_1']:.3f}")
    print(f"    counterfactual mean under T=0: {r_point['counterfactual_mean_0']:.3f}")
    print(f"  Bootstrap: ATE = {r_boot['ATE']:+.3f}   SE = {r_boot['SE']:.3f}"
          f"   95%CI = ({r_boot['CI95'][0]:+.3f}, {r_boot['CI95'][1]:+.3f})")
    print(f"  True ATE = 0.5\n")

    print("--- library cross-check (R stdReg::stdGlm, gfoRmula; Python zepid.G-formula) ---")

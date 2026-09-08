"""Friedman's H-statistic (Reference Sec 47.95).

Friedman & Popescu 2008 'Predictive learning via rule ensembles',
Ann Appl Stat 2(3). Model-agnostic interaction detection. For a
fitted black-box f and features j, k:

    H^2_{jk} = sum_i [ PD_{jk}(x_ij, x_ik) - PD_j(x_ij) - PD_k(x_ik) ]^2
              / sum_i PD_{jk}(x_ij, x_ik)^2

where PD_j(x) = E_{X_{-j}}[ f(X_{-j}, x) ] is the partial-
dependence function. H^2 in [0, 1]: 0 = purely additive, 1 =
interaction dominates.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.ensemble import GradientBoostingRegressor    # black-box model


def _pd(model, X, cols, grid_x=None):
    """Approximate partial dependence via Monte-Carlo integration."""
    Xc = X.copy()
    if len(cols) == 1:
        vals = X[:, cols[0]]
    else:
        vals = X[:, cols]
    preds = np.zeros(len(X))
    for i, v in enumerate(vals):
        Xc = X.copy()
        Xc[:, cols] = v
        preds[i] = float(model.predict(Xc).mean())
    return preds - preds.mean()


def h_stat(model, X, j, k):
    pd_j = _pd(model, X, [j])
    pd_k = _pd(model, X, [k])
    pd_jk = _pd(model, X, [j, k])
    num = np.sum((pd_jk - pd_j - pd_k) ** 2)
    den = np.sum(pd_jk ** 2) + 1e-12
    return float(np.sqrt(num / den))


if __name__ == "__main__":
    print("=== Friedman's H-statistic (Friedman-Popescu 2008) ===\n")
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 4))
    # Scenario A: purely additive
    y_add = 2 * X[:, 0] + X[:, 1] ** 2 - 0.5 * X[:, 2] + 0.2 * rng.normal(size=n)
    # Scenario B: strong (0, 2) interaction
    y_int = 2 * X[:, 0] + X[:, 1] ** 2 + 3 * X[:, 0] * X[:, 2] + 0.2 * rng.normal(size=n)

    for name, y in [("additive truth", y_add), ("with (0, 2) interaction", y_int)]:
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=0).fit(X, y)
        print(f"\n  {name}:")
        for j, k in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            H = h_stat(m, X[:200], j, k)     # subsample for speed
            marker = "  <-- true pair" if (name.startswith("with") and (j, k) == (0, 2)) else ""
            print(f"    H({j}, {k}) = {H:.3f}{marker}")

    print("\n--- library cross-check (iml::Interaction R; sklearn.inspection Python) ---")

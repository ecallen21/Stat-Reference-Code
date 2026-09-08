"""Weighted Quantile Sum (WQS) regression (Reference Sec 47.62).

Carrico, Gennings, Wheeler & Factor-Litvak 2015 'Characterization
of weighted quantile sum regression for highly correlated data in
a risk analysis setting', J Agric Biol Environ Stat 20(1).
Regresses outcome on a WEIGHTED INDEX of quantile-scored exposures:

    y = beta_0 + beta_1 * ( sum_i w_i * q_i(x_i) ) + Z gamma + eps
    sum_i w_i = 1,  w_i >= 0.

Simultaneously estimates the mixture EFFECT beta_1 and each
exposure's SHARE w_i. Directional constraint (positive or
negative) prevents sign cancellation across correlated exposures.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # bounded QP


def _weighted_index(w, Q):
    return Q @ w


def _lsq(y, WQ, Z):
    if Z is not None:
        X = np.column_stack([np.ones_like(y), WQ, Z])
    else:
        X = np.column_stack([np.ones_like(y), WQ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, float((resid ** 2).sum())


def wqs_fit(X, y, Z=None, n_quant=4, B=100, val_frac=0.4, direction="+", seed=0):
    """WQS with bootstrap of B training splits (Carrico et al 2015)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    # 1. Quantile-score every exposure into {0, ..., n_quant-1}
    Q = np.array([np.digitize(X[:, j], np.quantile(X[:, j],
                    np.linspace(0, 1, n_quant + 1)[1:-1])) for j in range(p)]).T.astype(float)

    weight_draws = []; beta_draws = []
    for b in range(B):
        idx = rng.permutation(n)
        train, val = idx[:int((1 - val_frac) * n)], idx[int((1 - val_frac) * n):]
        Qt, yt = Q[train], y[train]
        Zt = Z[train] if Z is not None else None

        def obj(w):
            WQ = _weighted_index(w, Qt)
            beta, ssr = _lsq(yt, WQ, Zt)
            return ssr

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * p
        res = minimize(obj, np.ones(p) / p, method="SLSQP",
                        bounds=bounds, constraints=cons,
                        options={"ftol": 1e-8, "maxiter": 200})
        w = res.x
        # Validate: fit beta_1 on validation set with this w
        WQv = _weighted_index(w, Q[val])
        beta_v, _ = _lsq(y[val], WQv, Z[val] if Z is not None else None)
        # Directional constraint
        if (direction == "+" and beta_v[1] > 0) or (direction == "-" and beta_v[1] < 0):
            weight_draws.append(w); beta_draws.append(beta_v[1])
    w_mean = np.mean(weight_draws, axis=0)
    beta_mean = float(np.mean(beta_draws))
    return {"w": w_mean, "beta1": beta_mean, "n_valid_boot": len(beta_draws)}


if __name__ == "__main__":
    print("=== Weighted Quantile Sum regression (Carrico et al 2015) ===\n")
    rng = np.random.default_rng(0)
    n, p = 400, 6
    # Correlated exposures with 2 truly bad ones (indices 0, 3)
    L = np.eye(p) + 0.4 * (rng.uniform(size=(p, p)) - 0.5)
    L = L @ L.T
    X = rng.multivariate_normal(mean=np.zeros(p), cov=L, size=n)
    X = X - X.min(0) + 1e-3     # positive
    true_w = np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0])
    q_scores = np.array([np.digitize(X[:, j], np.quantile(X[:, j], [0.25, 0.5, 0.75]))
                          for j in range(p)]).T.astype(float)
    y = 1.0 + 0.4 * (q_scores @ true_w) + 0.3 * rng.normal(size=n)

    fit = wqs_fit(X, y, direction="+", B=80)
    print(f"  True weights   (index 0 and 3): {true_w}")
    print(f"  Estimated w    (across {fit['n_valid_boot']}/80 bootstraps in valid direction):")
    for j in range(p):
        marker = "  <== TRUE" if true_w[j] > 0 else ""
        print(f"    w[{j}] = {fit['w'][j]:.3f}{marker}")
    print(f"  Estim beta1 (mixture effect) = {fit['beta1']:.3f}   (truth 0.400)")

    print("\n--- library cross-check (gWQS R; wqspy Python) ---")

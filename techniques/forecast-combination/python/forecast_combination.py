"""Forecast combination (Reference §13.30).

Combining multiple forecasts often produces a lower forecast error than
any single one -- a phenomenon so robust it is sometimes called the
'forecast combination puzzle': the equal-weighted average frequently beats
individually-fitted optimal weights.

Methods
    1. Simple average       -- take the mean of the forecasts.
    2. Trimmed average      -- drop the top-alpha and bottom-alpha forecasts.
    3. Bates-Granger (1969) -- inverse-variance weights from historical errors.
    4. Granger-Ramanathan   -- OLS regression of y on the forecasts (with or
       without constraint that weights sum to 1 and >= 0).

    5. Discounted weights   -- exponentially weight recent errors more heavily.

Rationale: uncorrelated forecast errors partially cancel; combining reduces
variance.  Correlated errors don't add much diversification -- select
combinations of DIFFERENT model classes.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def simple_average(F) -> np.ndarray:
    """Row-wise mean of a T x K forecast matrix."""
    return np.asarray(F, dtype=float).mean(axis=1)


def trimmed_average(F, trim: float = 0.2) -> np.ndarray:
    """Row-wise trimmed mean; drops the top and bottom `trim` fraction."""
    F = np.asarray(F, dtype=float); T, K = F.shape
    n_drop = int(round(K * trim))
    out = np.empty(T)
    for t in range(T):
        s = np.sort(F[t])
        out[t] = s[n_drop:K - n_drop].mean() if K - 2 * n_drop > 0 else s.mean()
    return out


def bates_granger(F_train, y_train, F_test) -> dict:
    """Inverse-variance weights from training errors."""
    F_train = np.asarray(F_train, dtype=float); y_train = np.asarray(y_train, dtype=float)
    errors = F_train - y_train[:, None]
    var_i = errors.var(axis=0, ddof=1)
    w = (1 / var_i) / (1 / var_i).sum()
    return {"weights": w, "combined": np.asarray(F_test, dtype=float) @ w,
            "method": "Bates-Granger inverse-variance weights"}


def granger_ramanathan(F_train, y_train, F_test, constrain: bool = False) -> dict:
    """OLS combination weights; optionally constrained to non-negative + sum-to-1."""
    F_train = np.asarray(F_train, dtype=float); y_train = np.asarray(y_train, dtype=float)
    if constrain:
        from scipy.optimize import minimize
        K = F_train.shape[1]
        def loss(w): return np.mean((F_train @ w - y_train) ** 2)
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * K
        res = minimize(loss, np.full(K, 1 / K), constraints=cons, bounds=bounds)
        w = res.x
    else:
        # OLS with an intercept
        X = np.column_stack([np.ones(len(y_train)), F_train])
        beta, *_ = np.linalg.lstsq(X, y_train, rcond=None)
        w = beta[1:]  # discard intercept for the "weights" summary
    return {"weights": w,
            "combined": (np.column_stack([np.ones(len(F_test)), np.asarray(F_test)]) @
                         np.concatenate([[0 if constrain else beta[0]], w]))
                        if not constrain else np.asarray(F_test) @ w,
            "constrained": constrain,
            "method": "Granger-Ramanathan OLS combining" + (" (constrained)" if constrain else "")}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 100; K = 5  # 5 forecasters over 100 periods
    true_y = np.cumsum(rng.normal(0, 1, T))
    # Simulate K biased noisy forecasters
    bias = rng.normal(0, 0.3, K)
    noise_sd = rng.uniform(0.5, 2.0, K)
    F = true_y[:, None] + bias + rng.normal(0, 1, (T, K)) * noise_sd
    # Split
    y_tr, y_te = true_y[:70], true_y[70:]
    F_tr, F_te = F[:70], F[70:]

    def mse(pred, obs): return float(np.mean((pred - obs) ** 2))

    print("=== Individual forecaster MSEs (test) ===")
    for k in range(K):
        print(f"  forecaster {k+1}: MSE = {mse(F_te[:, k], y_te):.3f}")

    print("\n=== Combinations (test MSE) ===")
    for name, pred in [("simple mean", simple_average(F_te)),
                        ("trimmed mean (20%)", trimmed_average(F_te, trim=0.2))]:
        print(f"  {name:20s}: MSE = {mse(pred, y_te):.3f}")
    bg = bates_granger(F_tr, y_tr, F_te)
    print(f"  {'Bates-Granger':20s}: MSE = {mse(bg['combined'], y_te):.3f}  weights = {bg['weights'].round(3)}")
    gr = granger_ramanathan(F_tr, y_tr, F_te)
    print(f"  {'Granger-Ramanathan':20s}: MSE = {mse(gr['combined'], y_te):.3f}  weights = {gr['weights'].round(3)}")
    grc = granger_ramanathan(F_tr, y_tr, F_te, constrain=True)
    print(f"  {'GR (constrained)':20s}: MSE = {mse(grc['combined'], y_te):.3f}  weights = {grc['weights'].round(3)}")

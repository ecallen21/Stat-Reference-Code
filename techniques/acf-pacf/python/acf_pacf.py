"""ACF, PACF, Ljung-Box, CCF, Mann-Kendall (Reference §13.1; also §13.9, §13.42, §13.48).

Time-series diagnostic building blocks:

- Sample AUTOCORRELATION function (ACF):
      rho_hat(k) = sum_{t = k+1}^n (x_t - xbar)(x_{t-k} - xbar) / sum_t (x_t - xbar)^2
  Range [-1, 1]. Bar-chart with 95% CI band = "correlogram."

- Sample PARTIAL AUTOCORRELATION function (PACF):
  Correlation between x_t and x_{t-k} after PARTIALLING OUT the effect of the
  intermediate lags. Computed by solving the Yule-Walker equations at each lag.
  Distinguishes AR(p) (PACF cuts off at lag p) from MA(q) (ACF cuts off at lag q).

- Ljung-Box portmanteau test (§13.42):
      Q = n(n+2) sum_{k=1}^h rho_hat(k)^2 / (n - k)   ~ chi^2_h under H0: no autocorr.
  Applied to model residuals; small p => leftover autocorrelation => model
  insufficient.

- Cross-correlation function (CCF, §13.48):
      rho_xy(k) = correlation between x_t and y_{t+k}.
  Look for peak lags to identify lead-lag structure.

- Mann-Kendall trend test (§13.9):
  Non-parametric test for monotonic trend.
      S = sum_{i<j} sign(x_j - x_i)
  Standardized S ~ N(0, 1) for large n (with variance formula that adjusts for
  ties).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def acf(x, nlags: int = 20) -> dict:
    """Sample ACF at lags 0..nlags with 95% confidence bounds (Bartlett)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = float((x * x).sum())
    rhos = np.zeros(nlags + 1)
    for k in range(nlags + 1):
        rhos[k] = float((x[k:] * x[: n - k]).sum() / denom) if denom > 0 else 0.0
    ci = 1.96 / math.sqrt(n)                   # Bartlett approximate 95% CI
    return {"lags": list(range(nlags + 1)),
            "acf": rhos.tolist(),
            "CI95": {"lower": -ci, "upper": ci},
            "n": int(n)}


def pacf(x, nlags: int = 20) -> dict:
    """Sample PACF via Yule-Walker at lags 1..nlags."""
    rhos = np.array(acf(x, nlags)["acf"])
    p = np.zeros(nlags + 1); p[0] = 1.0
    for k in range(1, nlags + 1):
        # Solve Yule-Walker Toeplitz system for order k
        R = np.array([[rhos[abs(i - j)] for j in range(k)] for i in range(k)])
        r = rhos[1: k + 1]
        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            phi = np.linalg.pinv(R) @ r
        p[k] = float(phi[-1])
    n = len(x)
    ci = 1.96 / math.sqrt(n)
    return {"lags": list(range(nlags + 1)),
            "pacf": p.tolist(),
            "CI95": {"lower": -ci, "upper": ci}}


def ljung_box(x_or_resid, lags=(5, 10, 20)) -> dict:
    """Ljung-Box Q statistic at one or more lag counts.

    Pass either the raw series (to test independence) or model residuals
    (to test remaining autocorrelation after fitting).
    """
    x = np.asarray(x_or_resid, dtype=float)
    n = len(x)
    rhos = np.array(acf(x, max(lags))["acf"])
    out = {}
    for h in lags:
        Q = n * (n + 2) * sum((rhos[k] ** 2) / (n - k) for k in range(1, h + 1))
        out[h] = {"Q": float(Q), "df": h,
                   "p_value": float(stats.chi2.sf(Q, h))}
    return {"portmanteau": out,
            "method": "Ljung-Box test at multiple lags"}


def ccf(x, y, nlags: int = 20) -> dict:
    """Cross-correlation function between two series at lags -nlags..+nlags."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x - x.mean(); y = y - y.mean()
    n = min(len(x), len(y))
    sx = math.sqrt((x * x).sum()); sy = math.sqrt((y * y).sum())
    denom = sx * sy
    lags = list(range(-nlags, nlags + 1))
    vals = []
    for k in lags:
        if k >= 0:
            v = (x[: n - k] * y[k:]).sum()          # x_t vs y_{t+k}
        else:
            v = (x[-k:] * y[: n + k]).sum()          # y leads x by |k|
        vals.append(float(v / denom) if denom > 0 else 0.0)
    ci = 1.96 / math.sqrt(n)
    return {"lags": lags, "ccf": vals,
            "CI95": {"lower": -ci, "upper": ci}}


def mann_kendall(x) -> dict:
    """Mann-Kendall monotone-trend test."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    S = 0
    for i in range(n - 1):
        S += int(np.sum(np.sign(x[i + 1:] - x[i])))
    # variance adjusted for ties
    _, counts = np.unique(x, return_counts=True)
    tied = counts[counts > 1]
    var_S = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in tied)) / 18
    if S > 0:   Z = (S - 1) / math.sqrt(var_S)
    elif S < 0: Z = (S + 1) / math.sqrt(var_S)
    else:       Z = 0.0
    return {"S": int(S), "Var_S": float(var_S), "Z": float(Z),
            "p_value_two_sided": float(2 * stats.norm.sf(abs(Z))),
            "trend": "increasing" if Z > 0 else "decreasing" if Z < 0 else "flat",
            "method": "Mann-Kendall trend test"}


def library_versions(x):
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, q_stat
    return {"statsmodels acf(5)": sm_acf(x, nlags=5).tolist(),
            "statsmodels pacf(5)": sm_pacf(x, nlags=5).tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 200
    # AR(1) with phi = 0.7 + gentle upward trend
    x = np.zeros(n); x[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    x = x + 0.03 * np.arange(n)                     # add trend

    print("=== ACF (first 5 lags) ===")
    a = acf(x, nlags=5)
    print(f"  lags: {a['lags']}")
    print(f"  acf : {[f'{r:+.3f}' for r in a['acf']]}")
    print(f"  95% CI: [{a['CI95']['lower']:.3f}, {a['CI95']['upper']:.3f}]")

    print("\n=== PACF (first 5 lags; AR(1) => big spike at lag 1, small after) ===")
    p = pacf(x, nlags=5)
    print(f"  pacf: {[f'{r:+.3f}' for r in p['pacf']]}")

    print("\n=== Ljung-Box (should reject; series has strong AR(1) autocorr) ===")
    lb = ljung_box(x, lags=(5, 10, 20))
    for k, v in lb["portmanteau"].items():
        print(f"  h={k}: Q={v['Q']:.2f}, p={v['p_value']:.4g}")

    print("\n=== Cross-correlation between x and lag-2 x (should peak at lag +2) ===")
    y = np.roll(x, 2) + rng.normal(0, 0.5, n)
    c = ccf(x, y, nlags=5)
    for lag, v in zip(c["lags"], c["ccf"]):
        star = " *" if abs(v) > c["CI95"]["upper"] else ""
        print(f"  lag {lag:+d}: {v:+.3f}{star}")

    print("\n=== Mann-Kendall (should be strongly increasing) ===")
    mk = mann_kendall(x); print(f"  {mk}")

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(x).items():
        print(f"  {k}: {v}")

"""Simple linear regression (Reference §5.1).

Model: y_i = beta_0 + beta_1 x_i + eps_i,  eps_i ~ N(0, sigma^2) iid.

From-scratch implementation of OLS with everything you'd get from
``lm(y ~ x)`` in R or ``statsmodels.OLS`` in Python: coefficient estimates,
standard errors, t-tests, F-test for the slope, R^2, residual SE, plus 95%
confidence and prediction intervals at new x values.

Closed-form OLS estimators
    beta_1_hat = sum((x - x_bar)(y - y_bar)) / sum((x - x_bar)^2) = cov / var
    beta_0_hat = y_bar - beta_1_hat * x_bar

Standard errors
    sigma_hat^2 = RSS / (n - 2)             unbiased residual variance
    SE(beta_1)  = sigma_hat / sqrt(sum((x - x_bar)^2))
    SE(beta_0)  = sigma_hat * sqrt(1/n + x_bar^2 / sum((x - x_bar)^2))

R^2 = 1 - RSS / TSS,  adjusted has no meaning for simple regression (use multiple).

Intervals at a new x0
    SE_mean(y | x0) = sigma_hat * sqrt(1/n + (x0 - x_bar)^2 / Sxx)
    SE_pred(y | x0) = sigma_hat * sqrt(1 + 1/n + (x0 - x_bar)^2 / Sxx)
"""
from __future__ import annotations

import math
from typing import Sequence

from scipy import stats


def _mean(x): return sum(x) / len(x)


def fit(x: Sequence[float], y: Sequence[float]) -> dict:
    """Fit y = b0 + b1 x via OLS. Returns coefficients, SEs, t / p, R^2, sigma_hat."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    n = len(x); mx = _mean(x); my = _mean(y)
    Sxx = sum((xi - mx) ** 2 for xi in x)
    Sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    Syy = sum((yi - my) ** 2 for yi in y)
    if Sxx == 0:
        raise ValueError("x has zero variance")
    b1 = Sxy / Sxx
    b0 = my - b1 * mx
    y_hat = [b0 + b1 * xi for xi in x]
    resid = [yi - yhi for yi, yhi in zip(y, y_hat)]
    rss = sum(r * r for r in resid)
    df = n - 2
    sigma2 = rss / df
    sigma = math.sqrt(sigma2)
    se_b1 = sigma / math.sqrt(Sxx)
    se_b0 = sigma * math.sqrt(1 / n + mx * mx / Sxx)
    t_b1 = b1 / se_b1
    t_b0 = b0 / se_b0
    p_b1 = 2 * stats.t.sf(abs(t_b1), df)
    p_b0 = 2 * stats.t.sf(abs(t_b0), df)
    r2 = 1 - rss / Syy if Syy > 0 else float("nan")
    return {"n": n, "x_bar": mx, "y_bar": my, "Sxx": Sxx, "Sxy": Sxy,
            "beta_0": b0, "beta_1": b1,
            "se_beta_0": se_b0, "se_beta_1": se_b1,
            "t_beta_0": t_b0, "t_beta_1": t_b1,
            "p_beta_0": float(p_b0), "p_beta_1": float(p_b1),
            "df_residual": df, "rss": rss, "sigma_hat": sigma,
            "r_squared": r2, "residuals": resid, "fitted": y_hat}


def predict_interval(fit_obj: dict, x0: float, conf: float = 0.95,
                     kind: str = "confidence") -> dict:
    """CI for the mean response (kind='confidence') or for a new observation ('prediction')."""
    b0 = fit_obj["beta_0"]; b1 = fit_obj["beta_1"]
    yhat = b0 + b1 * x0
    n = fit_obj["n"]; mx = fit_obj["x_bar"]; Sxx = fit_obj["Sxx"]
    sigma = fit_obj["sigma_hat"]; df = fit_obj["df_residual"]
    base = 1 / n + (x0 - mx) ** 2 / Sxx
    se = sigma * math.sqrt(base + (1.0 if kind == "prediction" else 0.0))
    tc = stats.t.ppf(0.5 + conf / 2, df)
    return {"x0": x0, "yhat": yhat, "se": se,
            "lower": yhat - tc * se, "upper": yhat + tc * se, "kind": kind}


def library_versions(x, y):
    """statsmodels OLS for a side-by-side check."""
    import statsmodels.api as sm
    import pandas as pd
    df = pd.DataFrame({"x": x, "y": y})
    res = sm.OLS(df["y"], sm.add_constant(df["x"])).fit()
    return {"statsmodels params": res.params.to_dict(),
            "statsmodels bse": res.bse.to_dict(),
            "statsmodels rsquared": float(res.rsquared),
            "statsmodels p-values": res.pvalues.to_dict()}


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 2.0 + 0.8 * x + rng.normal(0, 1.5, n)

    res = fit(x.tolist(), y.tolist())
    print("=== Fit ===")
    for k in ("beta_0", "beta_1", "se_beta_0", "se_beta_1",
              "t_beta_0", "t_beta_1", "p_beta_0", "p_beta_1",
              "sigma_hat", "r_squared", "df_residual"):
        print(f"  {k:12s}: {res[k]}")

    print("\n=== 95% CI for the mean response at x0 = 7 ===")
    print(predict_interval(res, 7.0, kind="confidence"))
    print("=== 95% prediction interval at x0 = 7 ===")
    print(predict_interval(res, 7.0, kind="prediction"))

    print("\n--- library (statsmodels) ---")
    for k, v in library_versions(x.tolist(), y.tolist()).items():
        print(f"  {k}: {v}")

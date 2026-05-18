"""GLM diagnostics (Reference §7.40, §7.41, §7.55).

Three complementary residual / fit-checking tools for GLMs:

1. PEARSON and DEVIANCE residuals.
   - r_P^i = (y_i - mu_i) / sqrt(V(mu_i))                         (Pearson)
   - r_D^i = sign(y_i - mu_i) * sqrt(d_i)                          (Deviance)
     where d_i is the unit deviance contribution.
   These mirror OLS residuals; large values flag bad-fitting points.

2. HOSMER-LEMESHOW goodness-of-fit (logistic specifically).
   Group fitted probabilities into g deciles, compare observed vs expected counts
   via a chi-square statistic with g-2 df. Sensitive to grouping choice and to
   ties; use with the caveat that for large n it rejects easily.

3. RANDOMIZED QUANTILE RESIDUALS (Dunn & Smyth 1996; the DHARMa-style fix).
   For DISCRETE outcomes, the usual residuals have a step structure that
   makes plots hard to read. RQRs put a uniform random "jitter" between
   F(y-1) and F(y) and map it to Phi^{-1} of the result. Under correct
   specification, RQRs are standard normal -- ANY pattern in their Q-Q plot
   or residuals-vs-fitted plot signals model misfit.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Callable    # stdlib: type hint meaning 'a function'

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def pearson_residuals(y, mu, variance_fn: Callable[[np.ndarray], np.ndarray]):
    """Pearson residuals: (y - mu) / sqrt(V(mu))."""
    y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
    v = variance_fn(mu)
    return (y - mu) / np.sqrt(np.clip(v, 1e-15, None))


def deviance_residuals_binomial(y, mu):
    """Deviance residuals for Bernoulli/binomial."""
    y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    d = 2 * (y * np.log(y / mu + 1e-12) + (1 - y) * np.log((1 - y) / (1 - mu) + 1e-12))
    return np.sign(y - mu) * np.sqrt(np.clip(d, 0, None))


def deviance_residuals_poisson(y, mu):
    """Deviance residuals for Poisson."""
    y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
    d = 2 * (np.where(y > 0, y * np.log(y / np.clip(mu, 1e-12, None)), 0) - (y - mu))
    return np.sign(y - mu) * np.sqrt(np.clip(d, 0, None))


def hosmer_lemeshow(y, p_hat, g: int = 10) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test for logistic regression."""
    y = np.asarray(y, dtype=float); p = np.asarray(p_hat, dtype=float)
    n = len(y)
    order = np.argsort(p)
    p_sorted = p[order]; y_sorted = y[order]
    # Decile breaks
    breaks = np.linspace(0, n, g + 1).astype(int)
    table = []
    chi2 = 0.0
    for k in range(g):
        sl = slice(breaks[k], breaks[k + 1])
        obs1 = float(y_sorted[sl].sum())
        exp1 = float(p_sorted[sl].sum())
        nk = breaks[k + 1] - breaks[k]
        obs0 = nk - obs1; exp0 = nk - exp1
        if exp1 > 0 and exp0 > 0:
            chi2 += (obs1 - exp1) ** 2 / exp1 + (obs0 - exp0) ** 2 / exp0
        table.append({"group": k + 1, "n": int(nk),
                      "obs_y1": int(obs1), "exp_y1": exp1,
                      "obs_y0": int(obs0), "exp_y0": exp0})
    df = g - 2
    p_val = float(stats.chi2.sf(chi2, df))
    return {"chi_square": chi2, "df": df, "p_value": p_val,
            "g": g, "table": table,
            "method": "Hosmer-Lemeshow"}


def randomized_quantile_residuals(y, mu, family: str, theta: float | None = None,
                                  rng=None):
    """Dunn-Smyth randomized quantile residuals (RQR) for discrete GLMs.

    family in {"poisson", "binomial", "nbinom"}.
    theta is the NB dispersion (only for "nbinom"). For "binomial" supply n_trials.
    """
    y = np.asarray(y, dtype=float); mu = np.asarray(mu, dtype=float)
    rng = rng or np.random.default_rng(0)
    if family == "poisson":
        F_lo = stats.poisson.cdf(y - 1, mu); F_hi = stats.poisson.cdf(y, mu)
    elif family == "binomial":  # mu is the probability; assume Bernoulli
        F_lo = np.where(y == 0, 0.0, 1 - mu)
        F_hi = np.where(y == 0, 1 - mu, 1.0)
    elif family == "nbinom":
        if theta is None:
            raise ValueError("theta required for negative binomial RQR")
        p_nb = theta / (theta + mu)
        F_lo = stats.nbinom.cdf(y - 1, theta, p_nb)
        F_hi = stats.nbinom.cdf(y, theta, p_nb)
    else:
        raise ValueError("family must be 'poisson', 'binomial', or 'nbinom'")
    u = rng.uniform(F_lo, F_hi)
    u = np.clip(u, 1e-15, 1 - 1e-15)
    return stats.norm.ppf(u)


def library_versions(y, mu):
    out = {}
    try:
        import statsmodels.api as sm
        out["statsmodels resid_pearson"] = "use res.resid_pearson on a fitted GLM"
        out["statsmodels resid_deviance"] = "use res.resid_deviance on a fitted GLM"
    except Exception:
        pass
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 300
    x = rng.normal(0, 1, n)
    eta = 0.2 + 0.8 * x
    mu = 1.0 / (1 + np.exp(-eta))
    y = (rng.uniform(0, 1, n) < mu).astype(int)

    # Pretend we fit logistic and got these mu's.
    p_hat = mu

    print("=== Pearson residuals (Bernoulli variance = mu(1-mu)) ===")
    rp = pearson_residuals(y, mu, lambda m: m * (1 - m))
    print(f"  mean = {rp.mean():+.4f}  sd = {rp.std():.4f}")

    print("\n=== Deviance residuals (binomial) ===")
    rd = deviance_residuals_binomial(y, mu)
    print(f"  mean = {rd.mean():+.4f}  sd = {rd.std():.4f}")

    print("\n=== Hosmer-Lemeshow test ===")
    hl = hosmer_lemeshow(y, p_hat, g=10)
    print(f"  chi^2({hl['df']}) = {hl['chi_square']:.4f}   p = {hl['p_value']:.4f}")

    # RQR example for Poisson
    print("\n=== Randomized quantile residuals (Poisson example) ===")
    mu_p = np.exp(0.5 + 0.6 * x)
    yp = rng.poisson(mu_p)
    rqr = randomized_quantile_residuals(yp, mu_p, family="poisson", rng=rng)
    print(f"  RQR: mean = {rqr.mean():+.4f}  sd = {rqr.std():.4f}   "
          "(should be ~ N(0,1) if model is correct)")
    # Shapiro-Wilk on a sub-sample for normality
    ks_stat = stats.kstest(rqr, "norm").statistic
    print(f"  K-S vs N(0,1): D = {ks_stat:.4f}")

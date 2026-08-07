"""Posterior predictive checks (Reference §14.19).

Diagnostic for Bayesian model fit.  Simulate REPLICATED datasets y_rep
from the posterior predictive:
    y_rep ~ p(y_rep | y) = integral p(y_rep | theta) p(theta | y) d theta

For each posterior draw theta^(s):
    1. Simulate a replicated dataset y_rep^(s) of the same size as y.
    2. Compute a summary statistic T(y_rep^(s)).

Compare the distribution of T(y_rep) to the observed T(y) via:
    - Visual overlay (histograms, densities).
    - Bayesian p-value: p_B = Pr(T(y_rep) >= T(y) | y).
      p_B ~ 0.5 = adequate; extreme (near 0 or 1) = misfit on T.

Common test statistics
    - mean, sd, min, max, quantiles.
    - number of zeros (for count models).
    - autocorrelation at lag 1 (for TS models).
    - proportion above a clinical threshold.

Warning: T should be selected to reflect a feature the model was NOT
designed to fit; testing T = mean on a Normal-mean model is nearly a
tautology and never rejects.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def ppc_bayesian_p(T_obs: float, T_rep) -> float:
    """Two-sided Bayesian p-value: min(p, 1 - p) x 2, one-sided uses direct comparison."""
    T_rep = np.asarray(T_rep, dtype=float)
    p = float((T_rep >= T_obs).mean())
    return p


def simulate_posterior_predictive_normal(mu_draws, sigma_draws, n_obs: int, seed: int = 0):
    """For a Normal likelihood, simulate y_rep given posterior draws."""
    rng = np.random.default_rng(seed)
    S = len(mu_draws)
    return np.column_stack([rng.normal(mu_draws, sigma_draws) for _ in range(n_obs)])


def posterior_predictive_check(y, y_rep_draws, statistics=("mean", "sd", "min", "max")) -> dict:
    """Run a suite of PP checks and return one row per statistic.

    y            : observed dataset (length N)
    y_rep_draws  : (S, N) array of simulated replicated datasets
    statistics   : names of statistics to compute
    """
    y = np.asarray(y, dtype=float); y_rep_draws = np.asarray(y_rep_draws, dtype=float)
    stat_fns = {"mean": np.mean, "sd": np.std, "min": np.min, "max": np.max,
                "median": np.median, "q10": lambda x: np.quantile(x, 0.10),
                "q90": lambda x: np.quantile(x, 0.90),
                "prop_pos": lambda x: (x > 0).mean(),
                "kurtosis": lambda x: float(np.mean((x - x.mean()) ** 4) / (x.var() ** 2) - 3)}
    rows = []
    for s in statistics:
        fn = stat_fns[s]
        T_obs = float(fn(y))
        T_rep = np.array([fn(row) for row in y_rep_draws])
        p_B = ppc_bayesian_p(T_obs, T_rep)
        rows.append({"statistic": s, "T_obs": T_obs,
                     "T_rep_mean": float(T_rep.mean()),
                     "T_rep_2.5": float(np.quantile(T_rep, 0.025)),
                     "T_rep_97.5": float(np.quantile(T_rep, 0.975)),
                     "bayesian_p": p_B,
                     "flag_misfit": abs(p_B - 0.5) > 0.45})
    return rows


def _print(rows, cols):
    def fmt(v):
        if isinstance(v, bool): return "yes" if v else "no"
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)
    w = {c: max(len(c), max(len(fmt(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(w[c]) for c in cols))
    print("  " + "  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(fmt(r[c]).ljust(w[c]) for c in cols))


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== Correctly-specified Normal model on Normal data ===")
    y = rng.normal(0, 1, 200)
    # Posterior for mu, sigma with diffuse priors
    n = len(y); ybar = y.mean(); s2 = y.var(ddof=1)
    S = 1500
    sig2 = 1 / rng.gamma(n / 2, 1 / (0.5 * (n - 1) * s2), S)
    mu = rng.normal(ybar, np.sqrt(sig2 / n), S)
    y_rep = np.array([rng.normal(mu[s], np.sqrt(sig2[s]), n) for s in range(S)])
    rows = posterior_predictive_check(y, y_rep, ("mean", "sd", "min", "max", "kurtosis"))
    _print(rows, ["statistic", "T_obs", "T_rep_mean", "T_rep_2.5", "T_rep_97.5", "bayesian_p", "flag_misfit"])

    print("\n=== Mis-specified Normal model on heavy-tailed (Cauchy) data ===")
    y = rng.standard_cauchy(200)
    y = y[np.abs(y) < 30]  # trim extreme tail so posterior is well-defined
    n = len(y); ybar = y.mean(); s2 = y.var(ddof=1)
    S = 1500
    sig2 = 1 / rng.gamma(n / 2, 1 / (0.5 * (n - 1) * s2), S)
    mu = rng.normal(ybar, np.sqrt(sig2 / n), S)
    y_rep = np.array([rng.normal(mu[s], np.sqrt(sig2[s]), n) for s in range(S)])
    rows = posterior_predictive_check(y, y_rep, ("mean", "sd", "min", "max", "kurtosis"))
    _print(rows, ["statistic", "T_obs", "T_rep_mean", "T_rep_2.5", "T_rep_97.5", "bayesian_p", "flag_misfit"])
    print("\n  Note: kurtosis, min, max should misfit under the Normal model since data are heavy-tailed.")

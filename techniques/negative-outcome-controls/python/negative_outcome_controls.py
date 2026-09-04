"""Negative outcome controls + empirical calibration (Reference Sec 43.9).

Schuemie et al. 2014.  Choose outcomes KNOWN not to be affected by
the exposure ("negative controls").  Under no residual confounding
they should give effect estimates centred on zero (log HR = 0).

Deviations from zero -> systematic error.  Fit an EMPIRICAL NULL
distribution from the negative-control effects, then re-calibrate
the p-values of the true-outcome analysis to correct for it.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats


def fit_null(logs, ses):
    """MLE of (mean, sd) of a systematic-error-inflated null."""
    logs = np.asarray(logs, dtype=float); ses = np.asarray(ses, dtype=float)
    # Under Schuemie: log_hr_i ~ N(mean_null, tau^2 + se_i^2)
    def nll(params):
        mu, log_tau = params
        var = np.exp(2 * log_tau) + ses ** 2
        return 0.5 * (np.log(2 * np.pi * var) + (logs - mu) ** 2 / var).sum()
    from scipy.optimize import minimize
    r = minimize(nll, x0=[0.0, np.log(np.std(logs) + 1e-3)],
                 method="Nelder-Mead", options={"xatol": 1e-6})
    mu, log_tau = r.x
    return {"mu_null": float(mu), "tau_null": float(np.exp(log_tau))}


def calibrate_p(log_hr, se, null_mu, null_tau):
    """Re-compute two-sided p under the empirical null."""
    z = (log_hr - null_mu) / np.sqrt(se ** 2 + null_tau ** 2)
    return 2 * stats.norm.sf(abs(z))


if __name__ == "__main__":
    print("=== Negative outcome controls + empirical calibration ===\n")
    rng = np.random.default_rng(0)
    # 40 negative controls: true log HR = 0, but systematic bias adds N(0.15, 0.10)
    n_nc = 40
    true_null_bias = 0.15
    ses = rng.uniform(0.08, 0.20, n_nc)
    logs = rng.normal(true_null_bias, 0.10 + ses)         # observed log HR
    fit = fit_null(logs, ses)
    print(f"  Empirical null estimated: mean = {fit['mu_null']:+.3f}"
          f"   sd = {fit['tau_null']:.3f}   (true bias mean = {true_null_bias})")

    # True outcome of interest: log HR = 0.4 (real effect), se = 0.12
    log_hr, se = 0.4, 0.12
    p_naive = 2 * stats.norm.sf(abs(log_hr / se))
    p_cal = calibrate_p(log_hr, se, fit["mu_null"], fit["tau_null"])
    print(f"\n  Real outcome: log HR = {log_hr}, SE = {se}")
    print(f"    Naive p = {p_naive:.3e}   (assumes no residual bias)")
    print(f"    Calibrated p = {p_cal:.3e}  (via empirical null)\n")

    print("--- library cross-check (R EmpiricalCalibration (OHDSI); Python custom + scipy) ---")

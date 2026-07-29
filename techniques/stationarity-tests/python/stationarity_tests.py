"""Stationarity tests: ADF, KPSS, Phillips-Perron + differencing (Reference §13.2, §13.8, §13.53).

Most classical time-series methods assume STATIONARITY:
    - constant mean
    - constant variance
    - autocovariance depending only on the lag, not on t

Three standard tests, each testing a DIFFERENT null:

    ADF (Augmented Dickey-Fuller): H0 = unit root (non-stationary)
        Small p => stationary.
        Regression:  Delta x_t = alpha + beta t + rho x_{t-1} + sum_i gamma_i Delta x_{t-i} + eps
        Test on rho == 0 vs rho < 0.

    KPSS: H0 = stationary (opposite null!)
        Small p => non-stationary.
        Tests residual sum-of-squares divided by long-run variance.

    Phillips-Perron: H0 = unit root, like ADF but nonparametric correction for
        autocorrelation and heteroscedasticity (rather than augmenting lags).

**How to reconcile disagreements** (§13.53):
    ADF rejects + KPSS not-reject -> stationary (agreement).
    ADF not-reject + KPSS rejects -> non-stationary (agreement).
    Both reject                     -> inconsistent; likely near-stationary or trending.
    Neither rejects                 -> inconclusive; large sample or heteroscedasticity issues.

    Practical rule: if non-stationary, DIFFERENCE once and re-test. Keep
    differencing until both tests agree on stationarity, or use trend/seasonal
    adjustment.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def difference(x, d: int = 1) -> np.ndarray:
    """Take d-th difference of x. Result length is len(x) - d."""
    x = np.asarray(x, dtype=float)
    for _ in range(d):
        x = np.diff(x)
    return x


def adf_test(x, max_lag: int = 5) -> dict:
    """Augmented Dickey-Fuller test via statsmodels (canonical implementation).

    From-scratch ADF would require MacKinnon critical values -- easier to
    delegate the p-value lookup to statsmodels.
    """
    from statsmodels.tsa.stattools import adfuller
    stat, p, usedlag, nobs, crit_vals, _ = adfuller(x, maxlag=max_lag, regression="ct")
    return {"statistic": float(stat), "p_value": float(p),
            "used_lag": int(usedlag), "n_obs": int(nobs),
            "critical_values": {k: float(v) for k, v in crit_vals.items()},
            "null_hypothesis": "unit root (non-stationary)",
            "small_p_means": "reject non-stationarity => stationary",
            "method": "Augmented Dickey-Fuller (constant + trend regression)"}


def kpss_test(x, regression: str = "c") -> dict:
    """KPSS test via statsmodels.

    regression: 'c' (constant / level-stationary) or 'ct' (trend-stationary).
    """
    from statsmodels.tsa.stattools import kpss
    stat, p, usedlag, crit_vals = kpss(x, regression=regression, nlags="auto")
    return {"statistic": float(stat), "p_value": float(p),
            "used_lag": int(usedlag),
            "critical_values": {k: float(v) for k, v in crit_vals.items()},
            "null_hypothesis": "stationary (opposite of ADF)",
            "small_p_means": "reject stationarity => non-stationary",
            "regression": regression,
            "method": "Kwiatkowski-Phillips-Schmidt-Shin test"}


def phillips_perron_test(x) -> dict:
    """Phillips-Perron test via arch package (or fallback message)."""
    try:
        from arch.unitroot import PhillipsPerron
        pp = PhillipsPerron(x)
        return {"statistic": float(pp.stat), "p_value": float(pp.pvalue),
                "null_hypothesis": "unit root (non-stationary)",
                "method": "Phillips-Perron (nonparametric ADF variant)"}
    except Exception as ex:
        return {"statistic": None, "p_value": None,
                "note": f"install 'arch' for PP test (pip install arch): {ex}",
                "method": "Phillips-Perron (unavailable)"}


def reconcile_adf_kpss(x, alpha: float = 0.05) -> dict:
    """Combined ADF + KPSS interpretation."""
    adf = adf_test(x)
    kps = kpss_test(x)
    adf_reject = adf["p_value"] < alpha
    kpss_reject = kps["p_value"] < alpha
    verdict = {
        (True, False):  "STATIONARY (ADF rejects unit root; KPSS fails to reject stationarity)",
        (False, True):  "NON-STATIONARY (both agree)",
        (True, True):   "INCONSISTENT (both reject) - likely near-stationary; try differencing",
        (False, False): "INCONCLUSIVE (neither rejects) - low power; try more data or a different test"
    }[(adf_reject, kpss_reject)]
    return {"adf": adf, "kpss": kps,
            "verdict": verdict, "alpha": alpha,
            "method": "ADF + KPSS combined stationarity assessment"}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 300
    # Random-walk-with-drift = non-stationary
    rw = np.cumsum(rng.normal(0.1, 1, n))
    # Stationary AR(1) around zero
    ar = np.zeros(n); ar[0] = rng.normal()
    for t in range(1, n):
        ar[t] = 0.5 * ar[t - 1] + rng.normal()

    print("=== Random walk (non-stationary; expect ADF NOT to reject, KPSS TO reject) ===")
    r = reconcile_adf_kpss(rw)
    print(f"  ADF  p = {r['adf']['p_value']:.4f}")
    print(f"  KPSS p = {r['kpss']['p_value']:.4f}")
    print(f"  Verdict: {r['verdict']}")

    print("\n=== After differencing the random walk ===")
    r = reconcile_adf_kpss(difference(rw))
    print(f"  ADF  p = {r['adf']['p_value']:.4f}")
    print(f"  KPSS p = {r['kpss']['p_value']:.4f}")
    print(f"  Verdict: {r['verdict']}")

    print("\n=== Stationary AR(1) (expect ADF TO reject, KPSS NOT to reject) ===")
    r = reconcile_adf_kpss(ar)
    print(f"  ADF  p = {r['adf']['p_value']:.4f}")
    print(f"  KPSS p = {r['kpss']['p_value']:.4f}")
    print(f"  Verdict: {r['verdict']}")

    print("\n=== Phillips-Perron on random walk (optional; needs 'arch') ===")
    pp = phillips_perron_test(rw)
    print(f"  {pp}")

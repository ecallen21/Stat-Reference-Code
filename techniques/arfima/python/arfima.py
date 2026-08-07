"""ARFIMA - Fractionally Integrated ARIMA (Reference §13.16).

Standard ARIMA(p, d, q) allows integer d (0, 1, 2, ...).  ARFIMA
generalizes to fractional d in (-0.5, 0.5), capturing LONG MEMORY:
    - d in (0, 0.5)   : long memory (positive slowly-decaying autocorrelations)
    - d = 0           : ordinary short memory
    - d in (-0.5, 0)  : anti-persistent (negative long-range dependence)

Fractional differencing (Granger-Joyeux-Hosking 1980):
    (1 - L)^d = sum_{k=0}^inf  binom(d, k) (-L)^k
    Coefficients pi_k = (-1)^k Gamma(d+1) / (Gamma(k+1) Gamma(d-k+1))

Estimation methods
    - Whittle spectral estimator: fit d via frequency-domain likelihood.
    - Geweke-Porter-Hudak (GPH) log-periodogram regression:
          log I(omega_j) = c - 2 d log|2 sin(omega_j / 2)| + eps
      Regress the low-frequency log-periodogram on the log-frequency term.

The demo below implements both a truncated fractional-differencing filter
and the GPH estimator for d.  Downstream ARMA(p, q) fit can be run on the
fractionally-differenced series with standard tools.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def fractional_diff_weights(d: float, n_lags: int = 200):
    """Coefficients pi_k of (1 - L)^d truncated at n_lags."""
    w = np.zeros(n_lags + 1); w[0] = 1.0
    for k in range(1, n_lags + 1):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w


def fractional_difference(x, d: float, n_lags: int = 200):
    """Apply truncated fractional differencing to a series."""
    x = np.asarray(x, dtype=float)
    w = fractional_diff_weights(d, n_lags)
    y = np.zeros_like(x)
    for t in range(len(x)):
        m = min(t + 1, n_lags + 1)
        y[t] = np.dot(w[:m], x[t::-1][:m])
    return y


def gph_estimator(x, m_frac: float = 0.5) -> dict:
    """Geweke-Porter-Hudak log-periodogram estimator of d."""
    x = np.asarray(x, dtype=float); x = x - x.mean(); T = len(x)
    Y = np.fft.rfft(x)
    I = (np.abs(Y) ** 2) / T
    freqs = np.fft.rfftfreq(T)[1:]
    I = I[1:]  # drop 0 frequency
    m = int(T ** m_frac)
    freqs = freqs[:m]; I = I[:m]
    x_reg = -2 * np.log(2 * np.sin(math.pi * freqs))
    y_reg = np.log(I + 1e-300)
    X_reg = np.column_stack([np.ones_like(x_reg), x_reg])
    beta, *_ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
    d_hat = float(beta[1])
    # Asymptotic SE for GPH
    se = math.sqrt(math.pi ** 2 / (24 * m))
    return {"d_hat": d_hat, "se_asymp": se,
            "z": d_hat / se, "p_value": float(2 * stats.norm.sf(abs(d_hat / se))),
            "m_bandwidth": int(m),
            "method": "Geweke-Porter-Hudak log-periodogram estimator of d"}


def simulate_arfima(n: int, d: float, seed: int = 0):
    """Simulate ARFIMA(0, d, 0) by fractional integration of white noise."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(size=n)
    w = fractional_diff_weights(-d, n_lags=min(n - 1, 300))
    x = np.zeros(n)
    for t in range(n):
        m = min(t + 1, len(w))
        x[t] = np.dot(w[:m], eps[t::-1][:m])
    return x


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T = 2000

    for d_true in (0.2, 0.35, 0.0):
        x = simulate_arfima(T, d=d_true, seed=int(d_true * 100))
        r = gph_estimator(x, m_frac=0.7)  # wider bandwidth for tighter SE
        print(f"=== ARFIMA(0, {d_true}, 0) simulated series, GPH estimate ===")
        print(f"  d_hat = {r['d_hat']:.3f}   SE = {r['se_asymp']:.3f}   p = {r['p_value']:.4f}")
        print(f"  bandwidth m = {r['m_bandwidth']}")

    # Fractional-differencing weights preview
    print("\n=== Fractional-diff weights (d = 0.4) ===")
    w = fractional_diff_weights(0.4, n_lags=8)
    print(f"  {w.round(4)}")

    print("\n--- library cross-check (statsmodels ARFIMA via ArmaProcess) ---")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        # Fit ordinary ARIMA(1,0,0) as sanity check on differenced series
        m = ARIMA(fractional_difference(x, d=0.35), order=(0, 0, 0)).fit()
        print(f"  statsmodels ARIMA fit on fractionally-differenced series: sigma^2 = {m.params[-1]:.3f}")
    except Exception as ex:
        print(f"  (statsmodels ARIMA not available or errored: {ex})")

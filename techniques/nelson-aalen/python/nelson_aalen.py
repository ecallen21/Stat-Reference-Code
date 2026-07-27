"""Nelson-Aalen cumulative hazard estimator (Reference §11.3; also covers §11.65).

Non-parametric estimator of the cumulative hazard H(t) from right-censored data:

    H_hat(t)  =  sum over event times t_j <= t of  d_j / n_j
        d_j = # events at t_j
        n_j = # at risk just before t_j

Related to Kaplan-Meier via:
    S_hat(t) ~ exp(-H_hat(t))    (exponential-of-cumulative-hazard identity;
                                   holds exactly in the limit of continuous time)

Variance (Aalen 1978):
    Var(H_hat(t)) = sum_{t_j <= t} d_j / n_j^2

Hazard rate estimation
----------------------
The instantaneous hazard h(t) is the derivative of H(t). Since H_hat is a
step function, we can smooth its increments to get a hazard-RATE estimate:

    h_hat(t)  =  (kernel smoother over event-time increments)

Simplest: box (uniform) kernel averaging jumps in a bandwidth b around t.
Gaussian, Epanechnikov, biweight are common alternatives (see
techniques/kernel-density-estimation).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def nelson_aalen(times, events) -> dict:
    """Nelson-Aalen cumulative hazard estimator + Aalen variance."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    order = np.argsort(times); t = times[order]; e = events[order]
    event_times = np.unique(t[e == 1])
    n = len(t)
    H = 0.0; var_H = 0.0
    H_series = []; var_series = []; at_risk = []; d_events = []
    for tj in event_times:
        n_j = int(np.sum(t >= tj))
        d_j = int(np.sum((t == tj) & (e == 1)))
        H += d_j / n_j if n_j > 0 else 0.0
        var_H += d_j / (n_j * n_j) if n_j > 0 else 0.0
        H_series.append(H); var_series.append(var_H)
        at_risk.append(n_j); d_events.append(d_j)
    H_series = np.array(H_series); var_series = np.array(var_series)
    z = stats.norm.ppf(0.975)
    # log-transformed CI for positive H
    with np.errstate(divide="ignore"):
        se_logH = np.where(H_series > 0, np.sqrt(var_series) / H_series, 0.0)
        ci_lo = H_series * np.exp(-z * se_logH)
        ci_hi = H_series * np.exp(z * se_logH)
    return {"event_times": event_times.tolist(),
            "n_at_risk": at_risk, "d_events": d_events,
            "H_hat": H_series.tolist(),
            "SE_H": np.sqrt(var_series).tolist(),
            "CI95_lower": ci_lo.tolist(),
            "CI95_upper": ci_hi.tolist(),
            "S_from_H": np.exp(-H_series).tolist(),
            "n_total": int(n), "n_events": int((e == 1).sum()),
            "method": "Nelson-Aalen cumulative hazard + log-CI"}


def hazard_rate_smoothed(times, events, grid, bandwidth: float,
                          kernel: str = "epanechnikov") -> dict:
    """Kernel-smoothed hazard-RATE estimate on a grid.

    Applies kernel smoothing to Nelson-Aalen INCREMENTS: h_hat(t) is essentially
    the (weighted) number of events per person-time in a bandwidth around t.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    grid = np.asarray(grid, dtype=float)
    order = np.argsort(times); t = times[order]; e = events[order]
    event_times = t[e == 1]
    # increments: delta H_j = 1 / n_j at each event time
    n_at_risk_at = np.array([np.sum(t >= tj) for tj in event_times])
    delta_H = 1.0 / n_at_risk_at

    def K(u):
        if kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0.0)
        if kernel == "gaussian":
            return np.exp(-0.5 * u ** 2) / math.sqrt(2 * math.pi)
        if kernel == "uniform":
            return np.where(np.abs(u) <= 1, 0.5, 0.0)
        raise ValueError("kernel must be epanechnikov / gaussian / uniform")

    h_hat = np.zeros_like(grid, dtype=float)
    for j, tj in enumerate(event_times):
        h_hat += K((grid - tj) / bandwidth) * delta_H[j] / bandwidth
    return {"grid": grid.tolist(), "hazard_rate": h_hat.tolist(),
            "bandwidth": bandwidth, "kernel": kernel,
            "method": "smoothed Nelson-Aalen hazard rate"}


def library_versions(times, events):
    try:
        from lifelines import NelsonAalenFitter
        naf = NelsonAalenFitter().fit(times, events)
        H = naf.cumulative_hazard_
        return {"lifelines NelsonAalen at demo times":
                {float(t): float(H.loc[t].iloc[0]) if t in H.index else None
                 for t in [1, 3, 5, 8]}}
    except Exception as ex:
        return {"lifelines (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 100
    true_lambda = 0.2                       # constant hazard = 0.2
    T_event = rng.exponential(1 / true_lambda, n)
    C_censor = rng.uniform(0, 10, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)

    na = nelson_aalen(times, events)
    print(f"=== Nelson-Aalen (n={na['n_total']}, events={na['n_events']}) ===")
    print(f"  H_hat at first 5 event times: {[f'{h:.4f}' for h in na['H_hat'][:5]]}")
    print(f"  S from H at first 5: {[f'{s:.4f}' for s in na['S_from_H'][:5]]}")

    # For exponential(lambda), H(t) = lambda * t, so H_hat / t should ~ lambda.
    t_mid = np.array(na["event_times"][20])
    print(f"\n  H_hat({t_mid:.2f}) = {na['H_hat'][20]:.4f}")
    print(f"  theoretical H(t) = lambda * t = {true_lambda * t_mid:.4f}")

    print("\n=== Smoothed hazard rate (should be ~ 0.2 constant) ===")
    hr = hazard_rate_smoothed(times, events, grid=[1, 2, 3, 4, 5, 6, 7],
                                bandwidth=1.5, kernel="epanechnikov")
    for tg, h in zip(hr["grid"], hr["hazard_rate"]):
        print(f"  t={tg}: h_hat={h:.4f}")

    print("\n--- library ---")
    for k, v in library_versions(times, events).items():
        print(f"  {k}: {v}")

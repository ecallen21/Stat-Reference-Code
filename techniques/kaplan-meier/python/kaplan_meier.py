"""Kaplan-Meier survival estimator (Reference §11.2).

Non-parametric estimator of the survival function S(t) from right-censored data:

    S_hat(t) = product over event times t_j <= t of (1 - d_j / n_j)
        d_j = # events at t_j
        n_j = # at risk just before t_j

Also covered here:
    - §11.1  Life tables            : same idea in fixed-interval grouping
    - §11.45 IPTW-KM adjusted curves : weighted at-risk / events by IPTW
    - §11.46 Median survival time   : smallest t with S_hat(t) <= 0.5
    - §11.61 Time origin            : caller-controlled -- see the demo comment
    - §11.68 KM plots with risk table: risk_table() helper

Variance & CI:
    Greenwood's formula:
        Var(S_hat(t)) = S_hat(t)^2 * sum_{t_j <= t} d_j / (n_j (n_j - d_j))
    Pointwise 95% CI on the LOG-LOG scale (better small-sample behavior than
    plain Wald on S):
        Let g(S) = log(-log S).  Then Var(g) = Var(S) / (S log S)^2.
        CI on g: g +/- z * sqrt(Var(g)); back-transform:
        CI on S: exp( -exp( g_upper ) ), exp( -exp( g_lower ) )

Median survival CI: Brookmeyer-Crowley (1982) inversion of a sign-based test.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def kaplan_meier(times, events, weights=None) -> dict:
    """Kaplan-Meier survival estimator with Greenwood variance + log-log CIs.

    Parameters
    ----------
    times : follow-up time for each subject.
    events : 1 = event, 0 = censored.
    weights : optional per-subject weights (e.g. IPTW for §11.45 adjusted KM).

    Returns
    -------
    dict with per-event-time arrays for t, n_at_risk, d_events, S_hat, and CI.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if weights is None:
        weights = np.ones_like(times)
    else:
        weights = np.asarray(weights, dtype=float)
    if len(times) != len(events) or len(times) != len(weights):
        raise ValueError("times, events, weights must have the same length")
    # Sort by time
    order = np.argsort(times)
    t = times[order]; e = events[order]; w = weights[order]

    # Distinct event times only (ties handled by aggregation)
    event_times = np.unique(t[e == 1])
    n = len(t)
    at_risk = []; d_events = []; cum_prod = 1.0
    S = []; var_S = []; sum_greenwood = 0.0
    for tj in event_times:
        # everyone still under observation at time tj (>= tj because at-risk is
        # "just before" and censored/events at tj count as at risk before)
        risk_mask = t >= tj
        n_j = float(w[risk_mask].sum())
        d_j = float(w[(t == tj) & (e == 1)].sum())
        if n_j <= 0: continue
        cum_prod *= (1 - d_j / n_j)
        S.append(cum_prod)
        # Greenwood
        denom = n_j * (n_j - d_j)
        if denom > 0:
            sum_greenwood += d_j / denom
        var_j = cum_prod * cum_prod * sum_greenwood
        var_S.append(var_j)
        at_risk.append(n_j); d_events.append(d_j)
    S = np.array(S); var_S = np.array(var_S)
    # Log-log CI
    z = stats.norm.ppf(0.975)
    ci_lower = np.empty_like(S); ci_upper = np.empty_like(S)
    for i, (s, v) in enumerate(zip(S, var_S)):
        if 0 < s < 1 and v > 0:
            g = math.log(-math.log(s))
            se_g = math.sqrt(v) / (s * abs(math.log(s)))
            hi_g = g + z * se_g; lo_g = g - z * se_g
            ci_upper[i] = math.exp(-math.exp(lo_g))
            ci_lower[i] = math.exp(-math.exp(hi_g))
        else:
            ci_lower[i] = ci_upper[i] = s
    return {"event_times": event_times.tolist(),
            "n_at_risk": at_risk,
            "d_events": d_events,
            "S_hat": S.tolist(),
            "SE_S": np.sqrt(var_S).tolist(),
            "CI95_lower": ci_lower.tolist(),
            "CI95_upper": ci_upper.tolist(),
            "n_total": int(n),
            "n_events": int((e == 1).sum()),
            "method": "Kaplan-Meier with Greenwood variance + log-log CIs"}


def median_survival(km) -> dict:
    """Median survival + Brookmeyer-Crowley 95% CI from a fitted KM object."""
    S = np.array(km["S_hat"]); t = np.array(km["event_times"])
    var_S = np.array(km["SE_S"]) ** 2
    below = np.where(S <= 0.5)[0]
    median = float(t[below[0]]) if len(below) else float("inf")
    # Brookmeyer-Crowley: solve for t such that
    #    (S_hat(t) - 0.5)^2 / Var(S_hat(t))  =  z_alpha^2
    z = stats.norm.ppf(0.975)
    # Search all event times
    inside = []
    for i, ti in enumerate(t):
        v = var_S[i]
        if v <= 0: continue
        if (S[i] - 0.5) ** 2 <= z * z * v:
            inside.append(float(ti))
    lo = min(inside) if inside else float("nan")
    hi = max(inside) if inside else float("nan")
    return {"median": median,
            "CI95_lower": lo, "CI95_upper": hi,
            "method": "Brookmeyer-Crowley (1982) inverted-test CI"}


def risk_table(times, events, grid, groups=None) -> dict:
    """Build a 'number at risk' table for KM plots (Reference §11.68).

    ``grid``   : sequence of times at which to report N at risk.
    ``groups`` : optional group labels; returns one row per group.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    grid = np.asarray(grid, dtype=float)
    if groups is None:
        groups = np.zeros_like(times, dtype=int)
    else:
        groups = np.asarray(groups)
    rows = []
    for g in np.unique(groups):
        mask = groups == g
        row = {"group": g.item() if hasattr(g, "item") else g}
        for tg in grid:
            row[float(tg)] = int(np.sum(times[mask] >= tg))
        rows.append(row)
    return {"risk_table": rows, "grid": grid.tolist()}


def library_versions(times, events):
    """lifelines KaplanMeierFitter -- gracefully skip if not installed."""
    try:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter().fit(times, events)
        S = kmf.survival_function_
        return {"lifelines KM at demo time points":
                {float(t): float(S.loc[t].iloc[0]) if t in S.index else None
                 for t in [1, 3, 5, 8]}}
    except Exception as ex:
        return {"lifelines (optional)": f"not available: {ex}"}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    # Time origin (Reference §11.61): here we take t = 0 as trial randomization.
    n = 100
    true_lambda = 0.15
    T_event = rng.exponential(1 / true_lambda, n)
    C_censor = rng.uniform(0, 12, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)

    km = kaplan_meier(times, events)
    print(f"=== Kaplan-Meier (n={km['n_total']}, events={km['n_events']}) ===")
    print(f"  distinct event times: {len(km['event_times'])}")
    print(f"  S_hat at t = {km['event_times'][:5]}: {[f'{s:.4f}' for s in km['S_hat'][:5]]}")
    print(f"  95% CI at first event: [{km['CI95_lower'][0]:.4f}, {km['CI95_upper'][0]:.4f}]")

    ms = median_survival(km)
    print(f"\n=== Median survival ===")
    print(f"  median = {ms['median']:.4f}   (theoretical = {math.log(2)/true_lambda:.4f} for exponential)")
    print(f"  Brookmeyer-Crowley 95% CI: [{ms['CI95_lower']:.4f}, {ms['CI95_upper']:.4f}]")

    print(f"\n=== Risk table at t = 0, 2, 4, 6, 8, 10 ===")
    rt = risk_table(times, events, grid=[0, 2, 4, 6, 8, 10])
    for row in rt["risk_table"]:
        print(f"  {row}")

    print("\n--- library ---")
    for k, v in library_versions(times, events).items():
        print(f"  {k}: {v}")

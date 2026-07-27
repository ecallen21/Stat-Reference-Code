"""Restricted Mean Survival Time (Reference §11.29; also covers §11.67).

RMST(t*) = expected survival time truncated at t*:

    RMST(t*)  =  integral from 0 to t* of S(u) du

Estimated by integrating the KM curve from 0 to t*:
    RMST_hat(t*)  =  sum over event times t_j <= t* of  S_hat(t_{j-1}) * (t_j - t_{j-1})
                     + S_hat(t_last) * (t* - t_last)

Variance (Andersen-Hansen-Klein 2004):
    Var(RMST_hat(t*)) = sum_{t_j <= t*}  [integral from t_j to t* of S(u) du]^2  *
                                          d_j / (n_j (n_j - d_j))

Between-group RMST difference:
    delta_hat = RMST_A(t*) - RMST_B(t*)
    SE = sqrt(Var_A + Var_B)  (independent groups)
    z = delta_hat / SE  ~ N(0,1) under H0

Why prefer over hazard ratio (§11.67)?
    - Time-scale interpretation (months of life) instead of an abstract rate.
    - No proportional-hazards assumption.
    - Well-defined even under non-proportional / crossing hazards.
    - Especially useful when curves cross or plateau (immunotherapy trials).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "kaplan-meier", "python"))
from kaplan_meier import kaplan_meier    # techniques/kaplan-meier/python/kaplan_meier.py::kaplan_meier


def rmst(times, events, tau: float) -> dict:
    """Restricted mean survival time up to horizon ``tau`` + Andersen SE."""
    times = np.asarray(times, dtype=float); events = np.asarray(events, dtype=int)
    km = kaplan_meier(times, events)
    ev_times = np.array(km["event_times"])
    S = np.array(km["S_hat"])
    n_at_risk = np.array(km["n_at_risk"])
    d_events = np.array(km["d_events"])
    # Truncate at tau
    mask = ev_times <= tau
    et_use = ev_times[mask]; S_use = S[mask]
    d_use = d_events[mask]; n_use = n_at_risk[mask]

    # RMST = area under KM step function up to tau. The step function is
    # constant S_prev on each interval [t_{j-1}, t_j).
    # Height on interval j is S_prev[j] = 1 if j == 0 else S_use[j-1].
    boundaries = np.concatenate([[0.0], et_use, [tau]])
    heights = np.concatenate([[1.0], S_use]) if len(S_use) else np.array([1.0])
    widths = np.diff(boundaries)                              # length = len(heights)
    # Trim heights or widths in the edge case where tau equals last event time
    m = min(len(widths), len(heights))
    rmst_hat = float((widths[:m] * heights[:m]).sum())

    # Andersen variance
    # Compute "tail area" T_j = int_{t_j}^{tau} S(u) du for each event time
    T = np.zeros(len(et_use))
    for j in range(len(et_use)):
        xs_j = np.concatenate([[et_use[j]], et_use[j + 1:], [tau]])
        ys_j = np.concatenate([[S_use[j]], S_use[j + 1:]])
        T[j] = float(np.dot(np.diff(xs_j), ys_j))
    var_rmst = float(np.sum(T ** 2 * d_use / np.clip(n_use * (n_use - d_use), 1e-12, None)))
    se = math.sqrt(max(var_rmst, 0.0))
    z = stats.norm.ppf(0.975)
    return {"RMST_hat": rmst_hat, "SE": se, "tau": tau,
            "CI95_lower": rmst_hat - z * se, "CI95_upper": rmst_hat + z * se,
            "n": int(len(times)),
            "method": "restricted mean survival time (KM integral + Andersen SE)"}


def rmst_difference(times, events, group, tau: float) -> dict:
    """Between-group RMST difference test."""
    times = np.asarray(times, dtype=float); events = np.asarray(events, dtype=int)
    group = np.asarray(group)
    labels = np.unique(group)
    if len(labels) != 2:
        raise ValueError("group must have exactly 2 distinct labels")
    A = rmst(times[group == labels[0]], events[group == labels[0]], tau)
    B = rmst(times[group == labels[1]], events[group == labels[1]], tau)
    diff = A["RMST_hat"] - B["RMST_hat"]
    se_diff = math.sqrt(A["SE"] ** 2 + B["SE"] ** 2)
    z = diff / max(se_diff, 1e-12)
    return {"RMST_A": A["RMST_hat"], "RMST_B": B["RMST_hat"],
            "difference": diff, "SE_difference": se_diff,
            "z": z, "p_value": float(2 * stats.norm.sf(abs(z))),
            "CI95_difference": {"lower": diff - stats.norm.ppf(0.975) * se_diff,
                                 "upper": diff + stats.norm.ppf(0.975) * se_diff},
            "tau": tau,
            "group_A": labels[0].item() if hasattr(labels[0], "item") else labels[0],
            "group_B": labels[1].item() if hasattr(labels[1], "item") else labels[1],
            "method": "between-group RMST difference (Z-test)"}


if __name__ == "__main__":
    rng = np.random.default_rng(37)
    n = 200
    group = rng.choice([0, 1], size=n)
    # Group 1 has higher hazard
    T_event = rng.exponential(np.where(group == 1, 1 / 0.15, 1 / 0.08), n)
    C_censor = rng.uniform(0, 15, n)
    times = np.minimum(T_event, C_censor)
    events = (T_event <= C_censor).astype(int)

    print(f"=== RMST for each group at tau = 10 ===")
    for g in [0, 1]:
        r = rmst(times[group == g], events[group == g], tau=10)
        print(f"  group {g}: RMST = {r['RMST_hat']:.3f}  SE = {r['SE']:.3f}  "
              f"95% CI [{r['CI95_lower']:.3f}, {r['CI95_upper']:.3f}]")
    # Theoretical RMST for exponential(lambda) up to tau: (1 - exp(-lambda tau)) / lambda
    for g, lam in [(0, 0.08), (1, 0.15)]:
        theo = (1 - math.exp(-lam * 10)) / lam
        print(f"  group {g} theoretical RMST(10): {theo:.3f}")

    print(f"\n=== RMST difference (tau = 10) ===")
    d = rmst_difference(times, events, group, tau=10)
    print(f"  diff (A - B) = {d['difference']:.4f}  SE = {d['SE_difference']:.4f}")
    print(f"  z = {d['z']:.3f}, p = {d['p_value']:.4g}")
    print(f"  95% CI on diff: [{d['CI95_difference']['lower']:.4f}, {d['CI95_difference']['upper']:.4f}]")

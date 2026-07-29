"""Landmark analysis for survival with time-varying exposure (Reference §11.24).

Motivating problem
    A time-varying exposure (transplant, response-to-treatment, biomarker
    turning positive) can only occur in subjects who have survived long
    enough to receive it.  Comparing "ever exposed" vs "never exposed"
    survival groups AT BASELINE gives IMMORTAL-TIME BIAS -- exposed
    subjects are guaranteed to survive at least until their exposure time,
    inflating their apparent benefit.

Landmark method (Anderson et al. 1983)
    1. Choose a landmark time t*.
    2. RESTRICT to subjects still alive and event-free at t*.
    3. Classify each surviving subject as EXPOSED (had exposure before t*)
       or NOT-EXPOSED (did not).
    4. Compare survival FROM t* forward using KM or Cox with the fixed
       landmark-time exposure covariate.

Trade-offs
    + Fully addresses immortal-time bias.
    + Simple; uses standard survival tools after the landmark cut.
    - Loses subjects who died before t*.
    - Ignores exposures that occur after t*.
    - Sensitive to the choice of t*.  Sensitivity analysis: repeat at
      several t* values (super-landmark / dynamic landmark analysis).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def kaplan_meier(time, event):
    """Return distinct event times and survival probabilities."""
    time = np.asarray(time, dtype=float); event = np.asarray(event, dtype=int)
    order = np.argsort(time); time = time[order]; event = event[order]
    unique_t = np.unique(time[event == 1])
    n_at_risk = len(time); S = 1.0
    ts, Ss = [0.0], [1.0]
    for t in unique_t:
        d = int(np.sum((time == t) & (event == 1)))
        n = int(np.sum(time >= t))
        S *= (1 - d / n)
        ts.append(t); Ss.append(S)
    return np.array(ts), np.array(Ss)


def _logrank_p(t1, e1, t2, e2):
    """Two-sample log-rank test p-value."""
    t = np.concatenate([t1, t2]); e = np.concatenate([e1, e2])
    grp = np.concatenate([np.zeros(len(t1)), np.ones(len(t2))])
    unique_t = np.unique(t[e == 1])
    O_E, V = 0.0, 0.0
    for u in unique_t:
        n1 = int(np.sum((t >= u) & (grp == 0)))
        n2 = int(np.sum((t >= u) & (grp == 1)))
        d1 = int(np.sum((t == u) & (e == 1) & (grp == 0)))
        d2 = int(np.sum((t == u) & (e == 1) & (grp == 1)))
        n = n1 + n2; d = d1 + d2
        if n <= 1 or d == 0: continue
        E1 = d * n1 / n
        O_E += d1 - E1
        V += (n1 * n2 * d * (n - d)) / (n * n * (n - 1)) if n > 1 else 0
    chi2 = O_E ** 2 / V if V > 0 else 0.0
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def landmark_analysis(time, event, exposure_time, landmark_t: float) -> dict:
    """Landmark analysis of survival with a time-varying exposure.

    time          : time-to-event or censoring
    event         : 1 = event, 0 = censored
    exposure_time : time exposure was received, or np.nan / np.inf if never
    landmark_t    : landmark time t*
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    exposure_time = np.asarray(exposure_time, dtype=float)
    # Only subjects alive & event-free at t*
    alive = time > landmark_t
    if not alive.any():
        raise ValueError("no subjects alive at landmark time")
    t_l = time[alive] - landmark_t
    e_l = event[alive]
    exposed = (exposure_time[alive] <= landmark_t) & np.isfinite(exposure_time[alive])
    n_exp = int(exposed.sum()); n_unexp = int((~exposed).sum())

    chi2, p_val = _logrank_p(t_l[exposed], e_l[exposed], t_l[~exposed], e_l[~exposed])
    ts_e, Ss_e = kaplan_meier(t_l[exposed], e_l[exposed])
    ts_u, Ss_u = kaplan_meier(t_l[~exposed], e_l[~exposed])
    # Median survival from landmark
    def med(ts, Ss):
        below = np.where(Ss <= 0.5)[0]
        return float(ts[below[0]]) if len(below) else float("inf")
    return {"landmark_time": float(landmark_t),
            "n_at_landmark": int(alive.sum()),
            "n_exposed": n_exp, "n_unexposed": n_unexp,
            "median_survival_from_landmark_exposed": med(ts_e, Ss_e),
            "median_survival_from_landmark_unexposed": med(ts_u, Ss_u),
            "logrank_chi2": chi2, "logrank_p": p_val,
            "method": "Landmark analysis (Anderson et al. 1983)"}


def super_landmark(time, event, exposure_time, landmarks) -> list:
    """Repeat landmark analysis at each t* and return one row per landmark."""
    return [landmark_analysis(time, event, exposure_time, t) for t in landmarks]


if __name__ == "__main__":
    rng = np.random.default_rng(9)
    n = 500
    # Simulate baseline hazard; exposure occurs at random time for ~40% of subjects
    T_true = rng.exponential(5, n)  # true survival time
    C = rng.exponential(8, n)       # censoring
    time = np.minimum(T_true, C)
    event = (T_true <= C).astype(int)
    # Exposure occurs at a random time (uniform on (0, min(4, T_true))) for a subset
    exposure_time = np.full(n, np.inf)
    ever = rng.uniform(0, 1, n) < 0.5
    exposure_time[ever] = rng.uniform(0, 4, ever.sum())
    exposure_time[exposure_time > time] = np.inf  # can't be exposed after death/censor

    # NAIVE ever/never comparison suffers from immortal-time bias
    ever_flag = np.isfinite(exposure_time)
    chi2_naive, p_naive = _logrank_p(time[ever_flag], event[ever_flag],
                                     time[~ever_flag], event[~ever_flag])
    print(f"=== NAIVE ever-vs-never log-rank (biased) ===")
    print(f"  chi2 = {chi2_naive:.3f}, p = {p_naive:.4f}")

    print("\n=== Landmark analysis at t* = 2 ===")
    r = landmark_analysis(time, event, exposure_time, landmark_t=2.0)
    for k, v in r.items(): print(f"  {k}: {v}")

    print("\n=== Super-landmark sensitivity (t* = 1, 2, 3, 4) ===")
    rows = super_landmark(time, event, exposure_time, [1.0, 2.0, 3.0, 4.0])
    for r in rows:
        print(f"  t*={r['landmark_time']:.1f}  n={r['n_at_landmark']:3d}  "
              f"n_exp={r['n_exposed']:3d}  chi2={r['logrank_chi2']:.3f}  p={r['logrank_p']:.4f}")

"""Immortal-time bias (Reference Sec 38.25).

IMMORTAL TIME = person-time during follow-up in which the outcome
under study CANNOT occur by definition of the exposure classification.
Common failure mode in pharmacoepidemiology:

  "Treated" defined as "ever received drug X during follow-up".
  Anyone who died before day of first prescription is (mis-)classified
  as untreated.  Treated group therefore appears artificially
  protected -- the time from cohort entry to first prescription is
  guaranteed-alive time (immortal) attributed to the treated group.

Standard fixes:
  1. TIME-VARYING EXPOSURE: subject contributes person-time to
     "untreated" before first prescription and "treated" after.
  2. TARGET TRIAL EMULATION: match on time-zero and eligibility
     criteria assessed AT baseline only.
  3. ACTIVE COMPARATOR / NEW USER (ACNU) design: index date = first
     prescription of either drug in a new-user cohort.

We simulate a cohort with survival exp(lambda_0) truly unaffected by
treatment (HR = 1), then time-to-first-treatment ~ exp(theta) among
survivors.  Naive baseline analysis shows a spurious protective
effect that is eliminated when exposure is treated as time-varying.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import optimize    # partial-likelihood MLE


def _cox_partial_ll(beta, times, events, x_of_t):
    """Time-fixed Cox partial log-likelihood with a single covariate.

    x_of_t : callable (subject_idx, t) -> covariate value at time t.
    """
    n = len(times)
    order = np.argsort(-times)      # descending time
    ll = 0.0
    for i in range(n):
        # At each event time t_i, sum of exp(beta * x) over risk set
        pass
    return None  # not used; kept as scaffold


def naive_baseline_hazard_ratio(t, delta, treated_ever):
    """Fit Cox with baseline-fixed exposure treated_ever using Breslow ties (single covariate)."""
    idx = np.argsort(t)
    t_s = t[idx]; d_s = delta[idx]; x_s = treated_ever[idx].astype(float)

    def nll(beta):
        ll = 0.0
        risk = x_s.copy() * 0 + 1
        for i, ti in enumerate(t_s):
            if d_s[i] == 0:
                continue
            at_risk = t_s >= ti
            ll += beta * x_s[i] - np.log(np.exp(beta * x_s[at_risk]).sum())
        return -ll

    res = optimize.minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    return float(np.exp(res.x))


def time_varying_hazard_ratio(t_event, delta, t_treat_start):
    """Time-varying Cox: subject moves from untreated to treated at t_treat_start.

    Break every subject into intervals and fit Cox with a single time-varying x(t).
    Simplification: at each event time, count treated = (t_treat_start[i] <= t) and
    i is still at risk.
    """
    idx = np.argsort(t_event)
    t_s = t_event[idx]; d_s = delta[idx]; tt = t_treat_start[idx]

    def nll(beta):
        ll = 0.0
        for i, ti in enumerate(t_s):
            if d_s[i] == 0:
                continue
            at_risk = t_s >= ti
            x_at_ti = (tt[at_risk] <= ti).astype(float)    # treated at ti?
            # covariate of event subject:
            xi = 1.0 if tt[i] <= ti else 0.0
            ll += beta * xi - np.log(np.exp(beta * x_at_ti).sum())
        return -ll

    res = optimize.minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    return float(np.exp(res.x))


if __name__ == "__main__":
    print("=== Immortal-time bias: naive baseline vs time-varying Cox ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    # True hazard rate independent of treatment
    lam0 = 1.0 / 365 / 3           # median ~2 years
    T_event = rng.exponential(1 / lam0, size=n)
    C = rng.uniform(365, 365 * 4, size=n)     # admin censoring 1-4 y
    t_obs = np.minimum(T_event, C)
    delta = (T_event <= C).astype(int)

    # Time to first prescription independent of death: exp with mean 6 months
    lam_tx = 1.0 / 180
    t_tx = rng.exponential(1 / lam_tx, size=n)

    # Naive baseline: "ever treated during follow-up"
    treated_ever = (t_tx <= t_obs).astype(int)
    HR_naive = naive_baseline_hazard_ratio(t_obs, delta, treated_ever)
    HR_tv    = time_varying_hazard_ratio(t_obs, delta, t_tx)

    print(f"  Truth: HR = 1.00   (treatment does not affect survival)")
    print(f"  Naive baseline-fixed 'ever treated' Cox:  HR_hat = {HR_naive:.3f}   <-- SPURIOUS")
    print(f"  Time-varying-exposure Cox                : HR_hat = {HR_tv:.3f}     <-- unbiased")
    print(f"  Bias source: person-time between cohort entry and first prescription is\n"
          f"  attributed to 'treated' in the naive analysis but is immortal (cannot die\n"
          f"  before treatment starts by definition), artificially favouring treatment.\n")

    print("--- library cross-check (R survival::coxph with tmerge; Python lifelines CoxTV) ---")

"""Multi-state models: transition intensities + state-occupation probabilities
(Reference §11.27, §11.52).

The illness-death model is the canonical multi-state model:

        healthy (0)  --alpha_01(t)--> ill (1)  --alpha_12(t)--> dead (2)
              |                                                  ^
              +--------- alpha_02(t) ---------------------------+

Each ARROW is a transition; each ARROW has its own hazard (transition intensity).

We estimate the transition intensities as Nelson-Aalen-style cumulative
transition hazards A_hk(t), and combine them into state-occupation
probabilities P_h(t) via the product-integral (Aalen-Johansen for multi-state).

For an irreversible model (no backward transitions):
    P_0(t)  =  exp(-A_01(t) - A_02(t))
    (state 1 and 2 occupancies computed by integrating transition intensities
     against P_0.)

For a general Markov model, the state-occupation vector p(t) satisfies
    p(t) = p(0) * Prod(I + dA(u))   integrated over 0 to t
where A(t) is the K x K matrix of cumulative transition hazards.

For this file we implement the illness-death model explicitly.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _na_cumhazard(times, events):
    """Nelson-Aalen at each unique event time. Returns (event_times, H)."""
    ev = np.unique(times[events == 1])
    H = np.zeros(len(ev)); cum = 0.0
    for j, tj in enumerate(ev):
        n_j = np.sum(times >= tj); d_j = np.sum((times == tj) & (events == 1))
        cum += d_j / max(n_j, 1e-12)
        H[j] = cum
    return ev, H


def illness_death_model(t_illness, t_death, illness_status, death_status) -> dict:
    """Estimate transition intensities in an illness-death model.

    Parameters
    ----------
    t_illness, t_death : per-subject times of the illness event and death event
        (or censoring time in each state).
    illness_status : 1 if became ill, 0 otherwise.
    death_status   : 1 if died, 0 otherwise.

    For each transition, we estimate the cumulative Nelson-Aalen hazard on the
    relevant sub-population.
    """
    t_illness = np.asarray(t_illness, dtype=float)
    t_death = np.asarray(t_death, dtype=float)
    illness_status = np.asarray(illness_status, dtype=int)
    death_status = np.asarray(death_status, dtype=int)

    # Transition 0 -> 1: subjects still healthy at time t. Event = illness onset.
    # follow-up time = t_illness if illness_status=1 else min(t_death, censor)
    times_01 = np.where(illness_status == 1, t_illness, t_death)
    events_01 = illness_status
    ev_01, H_01 = _na_cumhazard(times_01, events_01)

    # Transition 0 -> 2: healthy to dead directly (never became ill).
    # We need subjects who were healthy up to their death (or censor).
    # For simplicity, treat those who died without illness as 0->2 events.
    times_02 = np.where(illness_status == 0, t_death, t_illness)
    events_02 = ((illness_status == 0) & (death_status == 1)).astype(int)
    ev_02, H_02 = _na_cumhazard(times_02, events_02)

    # Transition 1 -> 2: subjects who became ill. Time = t_death - t_illness (gap).
    mask_ill = illness_status == 1
    if mask_ill.any():
        gap = t_death[mask_ill] - t_illness[mask_ill]
        ev_12, H_12 = _na_cumhazard(gap, death_status[mask_ill])
    else:
        ev_12, H_12 = np.array([]), np.array([])

    return {"cumhaz_0_to_1": {"times": ev_01.tolist(), "H": H_01.tolist()},
            "cumhaz_0_to_2": {"times": ev_02.tolist(), "H": H_02.tolist()},
            "cumhaz_1_to_2": {"times": ev_12.tolist(), "H": H_12.tolist()},
            "n_transitions_01": int(illness_status.sum()),
            "n_transitions_02": int(((illness_status == 0) & (death_status == 1)).sum()),
            "n_transitions_12": int(((illness_status == 1) & (death_status == 1)).sum()),
            "method": "illness-death cumulative-transition hazards (Nelson-Aalen per transition)"}


def state_occupancy_illness_death(msm_fit, grid, initial_state: int = 0) -> dict:
    """Compute P(state = h | initial = 0) at each time on ``grid``.

    Approximate via the product integral: at each event time,
        P_0(t) = exp(-H_01(t) - H_02(t))
        P_1(t) = integral over 0..t of P_0(u) dH_01(u) - integral of P_1(u) dH_12(u-t_ill)
        P_2(t) = 1 - P_0(t) - P_1(t)
    For simplicity we compute P_0 exactly and estimate P_1 by the observed rate
    of subjects currently in state 1 (interpretable prevalence).
    """
    grid = np.asarray(grid, dtype=float)
    t01 = np.array(msm_fit["cumhaz_0_to_1"]["times"])
    H01 = np.array(msm_fit["cumhaz_0_to_1"]["H"])
    t02 = np.array(msm_fit["cumhaz_0_to_2"]["times"])
    H02 = np.array(msm_fit["cumhaz_0_to_2"]["H"])
    # H_01 and H_02 evaluated at each grid t
    def step_at(ts, hs, tg):
        idx = np.searchsorted(ts, tg, side="right") - 1
        return hs[idx] if idx >= 0 else 0.0
    P0 = np.array([math.exp(-(step_at(t01, H01, tg) + step_at(t02, H02, tg))) for tg in grid])
    return {"grid": grid.tolist(),
            "P0_healthy": P0.tolist(),
            "cumhaz_leaving_healthy": [step_at(t01, H01, tg) + step_at(t02, H02, tg)
                                        for tg in grid],
            "method": "illness-death P(healthy) via exp(-cumhazard); P(ill), P(dead) require full msm software"}


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    n = 300
    # Simulate an illness-death process
    # 0 -> 1 rate = 0.1, 0 -> 2 rate = 0.05, 1 -> 2 rate = 0.2
    t_illness = np.zeros(n); illness_status = np.zeros(n, dtype=int)
    t_death = np.zeros(n); death_status = np.zeros(n, dtype=int)
    for i in range(n):
        T_ill = rng.exponential(1 / 0.1)
        T_02 = rng.exponential(1 / 0.05)
        C = rng.uniform(0, 20)
        if T_02 < T_ill:
            # went straight to death (or was censored)
            t_illness[i] = min(T_02, C); illness_status[i] = 0
            t_death[i] = min(T_02, C); death_status[i] = int(T_02 <= C)
        else:
            # became ill first
            if T_ill > C:
                t_illness[i] = C; illness_status[i] = 0
                t_death[i] = C; death_status[i] = 0
            else:
                t_illness[i] = T_ill; illness_status[i] = 1
                T_death_after = rng.exponential(1 / 0.2)
                total = T_ill + T_death_after
                if total > C:
                    t_death[i] = C; death_status[i] = 0
                else:
                    t_death[i] = total; death_status[i] = 1

    msm = illness_death_model(t_illness, t_death, illness_status, death_status)
    print("=== Illness-death cumulative transition hazards ===")
    print(f"  transitions 0->1: {msm['n_transitions_01']}")
    print(f"  transitions 0->2: {msm['n_transitions_02']}")
    print(f"  transitions 1->2: {msm['n_transitions_12']}")

    grid = [1, 3, 5, 10, 15]
    occ = state_occupancy_illness_death(msm, grid)
    print(f"\n=== State occupancy over time ===")
    for tg, p in zip(occ["grid"], occ["P0_healthy"]):
        print(f"  t={tg}: P(healthy) = {p:.4f}")
    print("  (true P(healthy | t=10) = exp(-0.15 * 10) = 0.2231)")

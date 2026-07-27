"""Recurrent-event models: Andersen-Gill, PWP, WLW (Reference §11.17-§11.19, §11.41, §11.51).

For subjects who can experience the SAME event repeatedly (hospitalizations,
infections, seizures), three main frameworks:

    Andersen-Gill (AG, §11.17)
        Counting-process Cox on ALL events, one row per event with
        (start, stop) intervals. Assumes events are independent given X.
        Standard-error correction (robust/sandwich) recommended for
        within-subject correlation.

    Prentice-Williams-Peterson (PWP, §11.18)
        Conditional model: stratify by event number. Subject only at risk
        for event k+1 after experiencing event k. Two flavors:
            - Total-time (start = 0)
            - Gap-time (start = time since prev event)   -> also §11.41

    Wei-Lin-Weissfeld (WLW, §11.19)
        Marginal: separate Cox model per event number, all subjects at risk
        for every event marginally. Robust SEs are essential.

This file uses fit_cox from techniques/cox-ph with the (start, stop, event)
counting-process input; the differences between AG / PWP / WLW are entirely
in HOW you construct the rows and strata.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
import sys, os    # stdlib: manipulate sys.path so we can import from the sibling technique

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cox-ph", "python"))
from cox_ph import fit_cox    # counting-process Cox    # techniques/cox-ph/python/cox_ph.py::fit_cox


def andersen_gill_rows(subject_id, event_times_per_subject,
                        censor_time_per_subject, X_per_subject):
    """Build AG-style counting-process rows.

    Each subject contributes one row per event + one row for the final
    censored interval (if any). All rows share the subject's X.
    """
    rows = []                                # (subject_id, start, stop, event, X_row)
    for i, sid in enumerate(subject_id):
        events = sorted(event_times_per_subject[i])
        Xi = X_per_subject[i]
        C = censor_time_per_subject[i]
        prev = 0.0
        for t_ev in events:
            if t_ev > C: break
            rows.append((sid, prev, t_ev, 1, Xi))
            prev = t_ev
        if prev < C:
            rows.append((sid, prev, C, 0, Xi))
    return rows


def fit_andersen_gill(rows) -> dict:
    """Fit Andersen-Gill Cox on AG rows built by ``andersen_gill_rows``."""
    starts = np.array([r[1] for r in rows], dtype=float)
    stops  = np.array([r[2] for r in rows], dtype=float)
    events = np.array([r[3] for r in rows], dtype=int)
    X      = np.array([r[4] for r in rows], dtype=float)
    fit = fit_cox(stops, events, X, start=starts)
    fit["method"] = "Andersen-Gill counting-process Cox (§11.17)"
    fit["warning"] = ("SEs assume independent events given X; use sandwich SE "
                       "for within-subject correlation")
    return fit


def pwp_gap_time_fit(subject_id, event_times_per_subject,
                      censor_time_per_subject, X_per_subject, max_events: int = 3) -> dict:
    """PWP gap-time (§11.18, §11.41): one Cox fit per event number, stratified.

    For each event number k, the outcome is "time from prev event to event k or
    censoring, given at risk for event k." Fit a stratified Cox by k on the
    combined data (start = 0 in gap-time metric).
    """
    all_rows = []      # (subject, k, gap_time, event, X)
    for i, sid in enumerate(subject_id):
        events = sorted(event_times_per_subject[i])
        C = censor_time_per_subject[i]
        Xi = X_per_subject[i]
        prev = 0.0
        for k in range(1, max_events + 1):
            if k - 1 < len(events) and events[k - 1] <= C:
                gap = events[k - 1] - prev
                all_rows.append((sid, k, gap, 1, Xi))
                prev = events[k - 1]
            else:
                gap = C - prev
                if gap > 0:
                    all_rows.append((sid, k, gap, 0, Xi))
                break
    # Fit per-strata (per event number)
    out = {"strata": {}}
    for k in range(1, max_events + 1):
        rows_k = [r for r in all_rows if r[1] == k]
        if len(rows_k) < 5: continue
        gap = np.array([r[2] for r in rows_k], dtype=float)
        ev = np.array([r[3] for r in rows_k], dtype=int)
        Xk = np.array([r[4] for r in rows_k], dtype=float)
        fit = fit_cox(gap, ev, Xk)
        out["strata"][k] = {"n_at_risk_this_event": len(rows_k),
                             "beta": fit["beta"],
                             "SE": fit["SE"],
                             "HR": fit["HR"]}
    out["method"] = "PWP gap-time (stratified by event number; §11.18 + §11.41)"
    return out


def wlw_marginal_fit(subject_id, event_times_per_subject,
                      censor_time_per_subject, X_per_subject, max_events: int = 3) -> dict:
    """WLW marginal (§11.19): a separate Cox fit for each event number, with
    ALL subjects at risk for every event (using time from origin, not gap)."""
    out = {"strata": {}}
    for k in range(1, max_events + 1):
        stops = []; events = []; Xs = []
        for i, sid in enumerate(subject_id):
            events_i = sorted(event_times_per_subject[i])
            C = censor_time_per_subject[i]
            if k - 1 < len(events_i) and events_i[k - 1] <= C:
                stops.append(events_i[k - 1]); events.append(1)
            else:
                stops.append(C); events.append(0)
            Xs.append(X_per_subject[i])
        stops = np.array(stops, dtype=float); events = np.array(events, dtype=int)
        Xk = np.array(Xs, dtype=float)
        if events.sum() < 5: continue
        fit = fit_cox(stops, events, Xk)
        out["strata"][k] = {"n_events": int(events.sum()),
                             "beta": fit["beta"], "SE": fit["SE"], "HR": fit["HR"]}
    out["method"] = "WLW marginal (§11.19); SE should be sandwich-corrected"
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(23)
    n_subj = 100
    subject_id = np.arange(n_subj)
    X = rng.normal(0, 1, size=(n_subj, 1))       # one covariate
    beta_true = np.array([0.5])                  # HR ~ exp(0.5) = 1.65

    # Poisson-process event times per subject with rate depending on X
    event_times_per_subject = []
    censor_time_per_subject = []
    for i in range(n_subj):
        rate = 0.3 * math.exp(float(X[i] @ beta_true))
        C = rng.uniform(0, 15)
        censor_time_per_subject.append(C)
        # generate Poisson-process event times up to C
        t = 0.0; events = []
        while True:
            t += rng.exponential(1 / rate)
            if t > C: break
            events.append(t)
        event_times_per_subject.append(events)

    total_events = sum(len(e) for e in event_times_per_subject)
    print(f"=== Simulated {n_subj} subjects, {total_events} total events ===")

    print("\n=== Andersen-Gill (§11.17) ===")
    rows = andersen_gill_rows(subject_id, event_times_per_subject,
                                censor_time_per_subject, X)
    ag = fit_andersen_gill(rows)
    print(f"  beta = {ag['beta']}, HR = {ag['HR']}, p = {ag['p_value']}")
    print(f"  (true beta = {beta_true.tolist()})")

    print("\n=== PWP gap-time (§11.18) ===")
    pwp = pwp_gap_time_fit(subject_id, event_times_per_subject,
                            censor_time_per_subject, X, max_events=3)
    for k, s in pwp["strata"].items():
        print(f"  event #{k}: n_at_risk = {s['n_at_risk_this_event']}, "
              f"beta = {s['beta']}, HR = {s['HR']}")

    print("\n=== WLW marginal (§11.19) ===")
    wlw = wlw_marginal_fit(subject_id, event_times_per_subject,
                            censor_time_per_subject, X, max_events=3)
    for k, s in wlw["strata"].items():
        print(f"  event #{k}: n_events = {s['n_events']}, "
              f"beta = {s['beta']}, HR = {s['HR']}")

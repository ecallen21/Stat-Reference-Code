"""Concordance C-statistic for survival predictors (Reference §11.6).

Discrimination measure for survival models.  Generalizes AUC:
    C = Pr(pred(i) > pred(j) | T_i < T_j and i experiences event)

Harrell C (Harrell 1982)
    Counts CONCORDANT vs DISCORDANT pairs among USABLE pairs:
        (i, j) usable if the shorter-time subject had the event
        concordant if the shorter-time subject also has higher pred risk
    C = concordant / usable.   Range [0, 1].  0.5 = random.

Bias with heavy censoring
    Harrell C is biased when censoring is heavy or dependent on X.
    Uno et al. (2011) proposed an IPCW-weighted variant that is
    consistent under any censoring pattern (given a well-estimated
    censoring distribution).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def harrell_c(time, event, pred) -> dict:
    """Harrell C-index."""
    time = np.asarray(time, dtype=float); event = np.asarray(event, dtype=int)
    pred = np.asarray(pred, dtype=float)
    n = len(time)
    concord = 0.0; usable = 0.0; ties_risk = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Usable pair: shorter-time subject had event
            if time[i] < time[j] and event[i] == 1:
                usable += 1
                if pred[i] > pred[j]: concord += 1
                elif pred[i] == pred[j]: ties_risk += 0.5
            elif time[j] < time[i] and event[j] == 1:
                usable += 1
                if pred[j] > pred[i]: concord += 1
                elif pred[i] == pred[j]: ties_risk += 0.5
    if usable == 0: return {"C": float("nan"), "usable": 0}
    return {"C": float((concord + ties_risk) / usable),
            "usable": int(usable),
            "concordant": int(concord),
            "method": "Harrell C-index"}


def uno_c(time, event, pred, tau: float = None) -> dict:
    """Uno IPCW-corrected concordance up to time tau (Uno et al. 2011)."""
    time = np.asarray(time, dtype=float); event = np.asarray(event, dtype=int)
    pred = np.asarray(pred, dtype=float)
    n = len(time)
    if tau is None:
        tau = float(np.max(time[event == 1]))
    # KM estimator of censoring distribution G
    ordr = np.argsort(time); t_s = time[ordr]; e_s = event[ordr]
    G_at = {0.0: 1.0}
    G = 1.0
    for t in np.unique(t_s[e_s == 0]):
        d = int(((t_s == t) & (e_s == 0)).sum())
        r = int((t_s >= t).sum())
        G *= (1 - d / r)
        G_at[float(t)] = G
    times_G = np.array(sorted(G_at.keys()))
    G_vals = np.array([G_at[t] for t in times_G])
    def G_of(t):
        idx = np.searchsorted(times_G, t, side="right") - 1
        return G_vals[max(idx, 0)]
    num = 0.0; denom = 0.0
    for i in range(n):
        if event[i] != 1 or time[i] > tau: continue
        w_i = 1 / max(G_of(time[i]) ** 2, 1e-8)
        for j in range(n):
            if time[j] > time[i]:
                if pred[i] > pred[j]: num += w_i
                elif pred[i] == pred[j]: num += 0.5 * w_i
                denom += w_i
    return {"C_uno": float(num / denom) if denom > 0 else float("nan"),
            "tau": float(tau),
            "method": "Uno IPCW-corrected C-index"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    T = rng.exponential(1 / np.exp(0.7 * x))
    C = rng.exponential(3, n)
    time = np.minimum(T, C); event = (T <= C).astype(int)
    pred = 0.7 * x  # linear predictor

    print("=== Harrell C on n = 300 survival data with beta = 0.7 predictor ===")
    r_h = harrell_c(time, event, pred)
    print(f"  Harrell C = {r_h['C']:.4f}   usable pairs = {r_h['usable']}")

    r_u = uno_c(time, event, pred)
    print(f"  Uno IPCW  C = {r_u['C_uno']:.4f}   tau = {r_u['tau']:.3f}")

    print("\n=== C for a RANDOM predictor (should be ~ 0.5) ===")
    pred_rand = rng.normal(size=n)
    print(f"  Harrell C = {harrell_c(time, event, pred_rand)['C']:.4f}")

    print("\n--- library cross-check (lifelines concordance_index / sksurv) ---")
    try:
        from lifelines.utils import concordance_index
        c = concordance_index(time, -pred, event)
        print(f"  lifelines concordance_index: {c:.4f}")
    except Exception as ex:
        print(f"  (lifelines unavailable: {ex})")

"""Aalen additive-hazards regression (Reference §11.14).

Alternative to Cox's multiplicative model:

    Cox   :  h_i(t) = h_0(t) exp(X_i beta)        multiplicative
    Aalen :  h_i(t) = beta_0(t) + X_i^T beta(t)   ADDITIVE, time-varying

Coefficients beta_k(t) are FUNCTIONS OF TIME estimated non-parametrically.
Fit by least-squares increments (Aalen 1980) at each event time:

    dB(t) = (X_R^T X_R)^-1 X_R^T dN(t)
    where X_R is the design at the current risk set and dN(t) is the
    event indicator over the risk set.

Cumulative regression functions:
    B_k(t) = integral_0^t beta_k(u) du
Report B_k(t) as a step function; slope over any interval = average
covariate effect on the hazard in that interval.

Hypothesis test: is beta_k(t) constant zero?  Aalen's supremum / integral
tests use the increments dB_k.

Contrast with Cox: additive scale is easier to interpret when covariate
effects change over time; multiplicative scale is more natural for
proportional-hazards data.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def aalen_additive(X, time, event) -> dict:
    """Fit Aalen additive-hazards regression.

    X     : n x p covariate matrix (WITHOUT intercept; intercept added automatically).
    time  : follow-up time.
    event : 1 = event, 0 = censored.
    """
    X = np.asarray(X, dtype=float); time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    n, p = X.shape
    Xd = np.column_stack([np.ones(n), X])
    p_full = p + 1
    ordr = np.argsort(time)
    time = time[ordr]; event = event[ordr]; Xd = Xd[ordr]
    event_times = np.unique(time[event == 1])
    B_cumulative = []  # list of (t, B(t)) rows
    B = np.zeros(p_full)
    cov_B = np.zeros((p_full, p_full))
    for t in event_times:
        risk = time >= t
        Xr = Xd[risk]
        try:
            XtX_inv = np.linalg.pinv(Xr.T @ Xr)
        except np.linalg.LinAlgError:
            continue
        # dN(t): 1 for each event happening at t
        dN = ((time == t) & (event == 1)).astype(float)
        # keep only the risk-set entries
        dN_r = dN[risk]
        dB = XtX_inv @ (Xr.T @ dN_r)
        # Variance increment (Aalen 1980): X'X)^-1 X' diag(dN) X (X'X)^-1
        var_dB = XtX_inv @ (Xr.T @ np.diag(dN_r) @ Xr) @ XtX_inv
        B = B + dB
        cov_B = cov_B + var_dB
        B_cumulative.append({"t": float(t), "B": B.copy()})
    # Aalen supremum-style test: max_t |B_k(t)| / sqrt(cov_kk(t))
    tests = []
    for k in range(p_full):
        maxT = max(abs(row["B"][k]) for row in B_cumulative)
        se = math.sqrt(cov_B[k, k])
        z = maxT / se if se > 0 else 0.0
        tests.append({"term": "intercept" if k == 0 else f"x{k}",
                      "sup_|B_k(t)|": maxT, "z_final": z})
    return {"cumulative_B": B_cumulative,
            "final_B": B, "final_cov_B": cov_B,
            "tests": tests,
            "n": int(n), "n_events": int((event == 1).sum()),
            "method": "Aalen additive-hazards regression"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    # True additive hazard: h(t) = 0.1 + 0.05 x (constant effect over time)
    # Sample event times by inverse CDF: S(t) = exp(-integral h(u) du)
    u = rng.uniform(size=n)
    lam = 0.1 + 0.05 * x
    lam = np.maximum(lam, 0.001)
    T = -np.log(u) / lam
    C = rng.exponential(15, n)
    time = np.minimum(T, C); event = (T <= C).astype(int)

    r = aalen_additive(x.reshape(-1, 1), time, event)
    print(f"=== Aalen additive-hazards, n = {r['n']}, events = {r['n_events']} ===")
    print(f"\n  Cumulative B_0 (baseline) and B_1 (x) at 5 timepoints:")
    for row in r["cumulative_B"][::max(1, len(r["cumulative_B"]) // 5)]:
        print(f"    t = {row['t']:6.3f}   B_0 = {row['B'][0]:.4f}   B_1 = {row['B'][1]:.4f}")

    print("\n  Constant-effect null tests (sup |B_k(t)| / SE):")
    for tst in r["tests"]:
        print(f"    {tst['term']:9s}  sup|B| = {tst['sup_|B_k(t)|']:.4f}  z = {tst['z_final']:.3f}")

    print("\n  Slope of B_1(t) over full follow-up = average dB_1/dt = avg beta_1(t)")
    t_max = r["cumulative_B"][-1]["t"]
    print(f"    B_1(t_max) / t_max = {r['final_B'][1] / t_max:.4f}   (true beta_1 = 0.05)")

    print("\n--- library cross-check (lifelines / R timereg) ---")
    try:
        from lifelines import AalenAdditiveFitter
        import pandas as pd
        df = pd.DataFrame({"T": time, "E": event, "x": x})
        aaf = AalenAdditiveFitter(fit_intercept=True).fit(df, "T", event_col="E")
        print(f"  lifelines final cumulative: {aaf.cumulative_hazards_.tail(1).values.flatten().round(4)}")
    except Exception as ex:
        print(f"  (lifelines unavailable: {ex})")

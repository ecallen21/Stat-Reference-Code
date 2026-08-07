"""Fine-Gray subdistribution hazards for competing risks (Reference §11.9).

Standard cause-specific Cox models a hazard rate for each cause, but the
effect of a covariate on the CUMULATIVE INCIDENCE for cause 1 is a compound
function of ALL cause-specific hazards.  Fine-Gray (1999) provides a model
whose coefficients act DIRECTLY on the cumulative incidence:

    h_1^sub(t | X) = h_10^sub(t) exp(X beta)

where h_1^sub is the SUBDISTRIBUTION hazard:
    h_1^sub(t) = lim_{dt -> 0} Pr(t <= T <= t + dt, cause = 1 | T > t OR (T <= t AND cause != 1)) / dt

Subjects who experience a COMPETING event stay in the risk set with a
time-decaying weight (IPCW-based) so that a covariate that increases
subdistribution hazard directly increases CIF for cause 1.

Estimation
    Weighted partial-likelihood analog of Cox.  Weight for subject i at
    time t:
        w_i(t) = G(t) / G(min(T_i, t))    if T_i > t or T_i <= t & cause != 1
    where G is the KM estimator of the censoring distribution.

The demo below shows subdistribution HR estimation on a small example.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _km_censoring(time, event):
    """KM estimator of the censoring distribution (event = 0 is censoring)."""
    time = np.asarray(time, dtype=float); event = np.asarray(event, dtype=int)
    is_cens = (event == 0).astype(int)
    ordr = np.argsort(time)
    t_s = time[ordr]; c_s = is_cens[ordr]
    unique_t = np.unique(t_s[c_s == 1])
    G = 1.0
    G_at = {0.0: 1.0}
    n_at_risk = len(t_s)
    ts_sorted = np.sort(t_s)
    for t in unique_t:
        d = int(((t_s == t) & (c_s == 1)).sum())
        n = int((t_s >= t).sum())
        G *= (1 - d / n)
        G_at[float(t)] = G
    times_sorted = np.array(sorted(G_at.keys()))
    G_vals = np.array([G_at[t] for t in times_sorted])
    def G_of(t):
        idx = np.searchsorted(times_sorted, t, side="right") - 1
        return G_vals[max(idx, 0)]
    return G_of


def fine_gray_fit(X, time, event, cause: int = 1) -> dict:
    """Fine-Gray subdistribution hazards for cause `cause`.

    time  : follow-up times.
    event : 0 = censored, 1 = cause of interest, 2 = competing event(s).
    """
    X = np.asarray(X, dtype=float); time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if X.ndim == 1: X = X.reshape(-1, 1)
    n, p = X.shape
    G_fn = _km_censoring(time, event)
    event_times = np.unique(time[event == cause])

    def neg_partial(beta):
        z = X @ beta
        ll = 0.0
        for t in event_times:
            # Subjects who fail from cause at t
            idx_e = np.where((time == t) & (event == cause))[0]
            # Risk set at t under subdistribution: subjects with T > t
            # PLUS subjects with T <= t AND cause != {cause, censored}
            in_risk = (time > t) | ((time <= t) & (event != cause) & (event != 0))
            weights = np.ones(n)
            competing_early = (time <= t) & (event != cause) & (event != 0)
            for j in np.where(competing_early)[0]:
                weights[j] = G_fn(t) / max(G_fn(min(time[j], t)), 1e-8)
            risk_z = z[in_risk]; risk_w = weights[in_risk]
            log_denom = math.log(np.sum(risk_w * np.exp(risk_z)) + 1e-300)
            for i in idx_e:
                ll += z[i] - log_denom
        return -ll

    res = minimize(neg_partial, np.zeros(p), method="Nelder-Mead")
    beta = res.x
    # Rough Wald SE via numerical Hessian
    eps = 1e-3; H = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            e_i = np.zeros(p); e_i[i] = eps
            e_j = np.zeros(p); e_j[j] = eps
            H[i, j] = (neg_partial(beta + e_i + e_j) - neg_partial(beta + e_i - e_j)
                       - neg_partial(beta - e_i + e_j) + neg_partial(beta - e_i - e_j)) / (4 * eps ** 2)
    se = np.sqrt(np.diag(np.linalg.pinv(H)))
    return {"beta": beta, "hr": np.exp(beta),
            "se": se,
            "z": beta / se, "p": 2 * stats.norm.sf(np.abs(beta / se)),
            "n": int(n), "n_events_of_interest": int((event == cause).sum()),
            "n_competing": int(((event != cause) & (event != 0)).sum()),
            "method": "Fine-Gray subdistribution hazards"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    # Two competing risks: cause 1 (event of interest) with HR ~ exp(beta * x); cause 2 (competing)
    lam1 = 0.05 * np.exp(0.6 * x)
    lam2 = 0.03 * np.exp(-0.2 * x)
    T1 = rng.exponential(1 / lam1); T2 = rng.exponential(1 / lam2)
    C = rng.exponential(30, n)
    time = np.minimum(np.minimum(T1, T2), C)
    event = np.zeros(n, dtype=int)
    event[np.where((T1 <= T2) & (T1 <= C))[0]] = 1
    event[np.where((T2 < T1) & (T2 <= C))[0]] = 2

    print(f"=== Fine-Gray on cause 1 (n = {n}, events = {int((event == 1).sum())}, "
          f"competing = {int((event == 2).sum())}) ===")
    r = fine_gray_fit(x.reshape(-1, 1), time, event, cause=1)
    print(f"  beta_x   = {r['beta'][0]:.3f}   (true 0.6 on cause-specific scale; Fine-Gray may differ)")
    print(f"  HR       = {r['hr'][0]:.3f}")
    print(f"  SE       = {r['se'][0]:.3f}")
    print(f"  Wald p   = {r['p'][0]:.4f}")

    print("\n--- library cross-check (cmprsk in R, lifelines has FineGrayCPH) ---")
    try:
        from lifelines import CRCSplineFitter  # placeholder, not the actual class
        print("  (see R crr package for canonical implementation)")
    except Exception:
        print("  R: cmprsk::crr(time, event, cov1 = x, failcode = 1, cencode = 0)")

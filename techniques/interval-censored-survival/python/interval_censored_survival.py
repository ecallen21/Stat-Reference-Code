"""Interval-censored survival (Reference §11.20).

Each subject's event time T_i is known only to fall in an interval (L_i, R_i]:
    L_i = last visit at which subject was still event-free
    R_i = first visit at which the event was observed
Left-censored: L_i = 0.  Right-censored: R_i = infinity.

Naive workarounds (impute midpoint, use L_i, use R_i) all bias the survival
estimate.  Correct treatment:

Turnbull NPMLE (Turnbull 1976)
    Nonparametric MLE of S(t) that respects the interval structure.
    Uses SELF-CONSISTENCY / EM iteration on the disjoint "Turnbull intervals"
    -- the atoms at which S can drop -- found from the intersection graph
    of the observed (L_i, R_i].

Parametric alternatives
    Weibull / log-normal / log-logistic MLE with the likelihood
        L_i(theta) = S(L_i; theta) - S(R_i; theta).
    Use when a smooth parametric form is credible.

Regression (interval-censored Cox / AFT) is a further extension; the standard
implementation is icenReg::ic_par / ic_sp (R).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _turnbull_intervals(L, R):
    """Find the disjoint 'Turnbull intervals' [q_j, p_j] where S can drop."""
    left_pts = np.unique(L)
    right_pts = np.unique(R[np.isfinite(R)])
    # Candidate atom endpoints: all L's and all R's
    all_L = np.sort(np.unique(L))
    all_R = np.sort(np.unique(R))
    # Turnbull intervals: for each pair (q, p) where q in {L_i} and p in {R_j}
    # and no other L or R lies strictly between them, S may drop.
    atoms = []
    for q in all_L:
        candidates = all_R[all_R >= q]
        if len(candidates) == 0: continue
        p = candidates[0]  # smallest R >= q
        # Ensure no other L strictly between q and p
        between = all_L[(all_L > q) & (all_L < p)]
        if len(between) == 0:
            atoms.append((q, p))
    # Deduplicate
    seen = set(); out = []
    for a in atoms:
        if a not in seen:
            seen.add(a); out.append(a)
    return out


def turnbull_npmle(L, R, max_iter: int = 500, tol: float = 1e-8) -> dict:
    """Turnbull NPMLE of the survival function from interval-censored data.

    L, R : arrays of interval endpoints.  Use R = np.inf for right-censored.
    """
    L = np.asarray(L, dtype=float); R = np.asarray(R, dtype=float)
    n = len(L)
    intervals = _turnbull_intervals(L, R)
    if not intervals:
        raise RuntimeError("no Turnbull intervals found")
    m = len(intervals)
    # A[i, j] = 1 if interval j is contained in observed (L_i, R_i]
    A = np.zeros((n, m))
    for j, (q, p) in enumerate(intervals):
        A[:, j] = (q >= L) & (p <= R)
    if (A.sum(1) == 0).any():
        # Numerical fallback: also include intervals that partially overlap
        for j, (q, p) in enumerate(intervals):
            mask = (A[:, j] == 0) & (L <= p) & (q <= R)
            A[mask, j] = 1
    p_j = np.ones(m) / m
    for it in range(max_iter):
        p_ij = A * p_j
        p_ij /= p_ij.sum(1, keepdims=True)
        p_new = p_ij.sum(0) / n
        if np.max(np.abs(p_new - p_j)) < tol: break
        p_j = p_new
    # Build the step-function survival estimate
    right_ends = np.array([p for _, p in intervals])
    order = np.argsort(right_ends)
    right_ends = right_ends[order]; p_j = p_j[order]
    S = 1 - np.cumsum(p_j)
    return {"time_grid": right_ends, "S_estimate": S,
            "interval_masses": p_j, "n_intervals": int(m),
            "n_subjects": int(n), "iterations": int(it + 1),
            "method": "Turnbull NPMLE (self-consistency EM)"}


def weibull_interval_mle(L, R) -> dict:
    """Parametric Weibull MLE on interval-censored data.

    S(t) = exp(-(t/lambda)^k)
    L_i(theta) = S(L_i) - S(R_i)
    """
    from scipy.optimize import minimize
    L = np.asarray(L, dtype=float); R = np.asarray(R, dtype=float)
    def neg_ll(theta):
        log_lam, log_k = theta
        lam = math.exp(log_lam); k = math.exp(log_k)
        S_L = np.exp(-((L + 1e-12) / lam) ** k) if lam > 0 else np.ones_like(L)
        S_L = np.where(L <= 0, 1.0, np.exp(-(L / lam) ** k))
        S_R = np.where(np.isinf(R), 0.0, np.exp(-(R / lam) ** k))
        prob = S_L - S_R
        prob = np.clip(prob, 1e-12, None)
        return -np.sum(np.log(prob))
    res = minimize(neg_ll, [math.log(np.median(L[L > 0]) if (L > 0).any() else 1.0), 0.0], method="Nelder-Mead")
    lam, k = math.exp(res.x[0]), math.exp(res.x[1])
    return {"scale_lambda": float(lam), "shape_k": float(k),
            "neg_log_lik": float(res.fun),
            "median_time": float(lam * math.log(2) ** (1 / k)),
            "method": "Weibull MLE on interval-censored data"}


if __name__ == "__main__":
    rng = np.random.default_rng(6)
    n = 400
    T_true = rng.weibull(1.6, n) * 5  # true event times
    # Observe subjects at random visits; L = last visit event-free, R = first at which event seen
    visit_grid = np.arange(0, 15, 1.0)
    L = np.zeros(n); R = np.full(n, np.inf)
    for i in range(n):
        # Each subject has visits at a jittered subset of visit_grid
        visits = np.sort(rng.uniform(visit_grid[:-1], visit_grid[1:]))
        before = visits[visits < T_true[i]]
        after = visits[visits >= T_true[i]]
        L[i] = before[-1] if len(before) else 0.0
        R[i] = after[0] if len(after) else np.inf

    print("=== Turnbull NPMLE ===")
    r = turnbull_npmle(L, R)
    print(f"  n intervals = {r['n_intervals']}, EM iterations = {r['iterations']}")
    print("  time  S(t)")
    for t, s in list(zip(r["time_grid"], r["S_estimate"]))[:10]:
        print(f"  {t:5.2f}  {s:.4f}")

    print("\n=== Weibull MLE on interval-censored data ===")
    w = weibull_interval_mle(L, R)
    print(f"  lambda = {w['scale_lambda']:.3f}, k = {w['shape_k']:.3f}, median = {w['median_time']:.3f}")
    print(f"  true DGP: k = 1.6, lambda = 5")

    print("\n--- library cross-check (lifelines KaplanMeierFitter on midpoint - naive) ---")
    try:
        from lifelines import KaplanMeierFitter
        mid = np.where(np.isinf(R), L + 1.0, (L + R) / 2)
        event = (~np.isinf(R)).astype(int)
        kmf = KaplanMeierFitter().fit(mid, event)
        print(f"  naive-midpoint KM median: {kmf.median_survival_time_:.3f} (biased)")
    except Exception as ex:
        print(f"  (lifelines not available: {ex})")

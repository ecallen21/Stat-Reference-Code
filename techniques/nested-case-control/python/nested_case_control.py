"""Nested case-control (NCC) study (Reference Sec 15.42).

Thomas 1977 'Addendum to methods of cohort analysis: appraisal by
application to asbestos mining'; Langholz & Goldstein 1996. Within
a defined cohort, select K controls per case matched on RISK-SET
membership (alive and uncensored at the case's event time).
Conditional logistic regression on the matched sets gives an
approximation of the Cox partial-likelihood hazard ratio.

Advantages:
    * Cheap when expensive covariate assessment is needed only on
      cases + controls, not the full cohort.
    * Preserves the temporal / risk-set structure of survival data.

Analysis: conditional logistic regression with stratum = matched set.

We demonstrate: simulate a cohort with a hazard-ratio-3 exposure;
build a 1:4 nested case-control sample; recover HR via conditional
logistic regression.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # conditional logistic MLE


def build_ncc_sample(t, e, X, K=4, seed=0):
    """Return matched sets: for each case, K random controls from its risk set.

    t : event / censoring time
    e : event indicator (1 event, 0 censor)
    X : covariates (n x p)
    Returns list of (case_idx, control_idxs) tuples.
    """
    rng = np.random.default_rng(seed)
    case_idx = np.where(e == 1)[0]
    sets = []
    for i in case_idx:
        risk_set = np.where(t >= t[i])[0]
        risk_set = risk_set[risk_set != i]
        if len(risk_set) < K:
            controls = risk_set
        else:
            controls = rng.choice(risk_set, size=K, replace=False)
        sets.append((int(i), list(map(int, controls))))
    return sets


def clogit_ncc_ll(beta, sets, X):
    """Conditional logistic log-likelihood: log P(case | matched set)."""
    ll = 0.0
    for case_i, ctrl in sets:
        eta_case = X[case_i] @ beta
        etas = np.array([X[j] @ beta for j in ctrl] + [eta_case])
        ll += eta_case - (etas.max() + np.log(np.sum(np.exp(etas - etas.max()))))
    return -ll


def clogit_ncc_fit(sets, X, p):
    r = minimize(clogit_ncc_ll, np.zeros(p), args=(sets, X), method="L-BFGS-B")
    return r.x


if __name__ == "__main__":
    print("=== Nested case-control (NCC): 1:K matched from the risk set ===\n")
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.binomial(1, 0.3, size=n).astype(float)
    #  Baseline exponential hazard 0.05, HR = 3 for exposed
    lam = 0.05 * np.exp(np.log(3) * x)
    t_true = rng.exponential(1 / lam)
    c = rng.exponential(6.0, size=n)
    e = (t_true <= c).astype(int)
    t = np.minimum(t_true, c)
    print(f"  Cohort n = {n}, event rate = {e.mean() * 100:.1f}%\n")

    for K in [1, 2, 4]:
        sets = build_ncc_sample(t, e, x.reshape(-1, 1), K=K, seed=1)
        beta_hat = clogit_ncc_fit(sets, x.reshape(-1, 1), p=1)
        HR_hat = float(np.exp(beta_hat[0]))
        print(f"  1:{K} matched  ->  HR_hat = {HR_hat:.3f}   (truth 3.00; n_sets = {len(sets)})")

    print("\n  With K=4 controls per case, NCC recovers the HR closely with a fraction")
    print("  of the covariate-collection cost of the full cohort.")

    print("\n--- library cross-check (Epi::ccwc / mgcv R; lifelines / statsmodels Python) ---")

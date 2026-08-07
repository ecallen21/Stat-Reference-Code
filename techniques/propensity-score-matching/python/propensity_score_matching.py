"""Propensity Score Matching (Reference §15.6; Rosenbaum-Rubin 1983).

Observational data: treatment assignment T is confounded by observed X.
Estimate the average treatment effect on the treated (ATT):

    ATT = E[Y(1) - Y(0) | T = 1]

Propensity score
    e(x) = Pr(T = 1 | X = x)
Rosenbaum-Rubin (1983): if strong ignorability holds
    Y(0), Y(1) independent of T | X
then it also holds when conditioning on e(X) instead of the full X.

Matching procedure (1:1 nearest-neighbor, with replacement)
    1. Fit logistic regression of T on X -> e_hat(x).
    2. For each treated i, find control j* minimizing |e_hat(x_i) - e_hat(x_j)|.
    3. ATT_hat = mean(Y_i - Y_{j*(i)}).

Balance diagnostics
    Standardized mean difference (SMD) per covariate:
        SMD_k = (mean_treated - mean_control) / sqrt((var_treated + var_control) / 2)
    Rule of thumb: |SMD| < 0.1 = good balance.

Extensions: 1:M matching, caliper, kernel matching, coarsened exact matching (CEM),
optimal matching, entropy balancing.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def estimate_propensity(X, T) -> np.ndarray:
    """Logistic-regression propensity score, ~1 / (1 + exp(-X beta))."""
    X = np.asarray(X, dtype=float); T = np.asarray(T, dtype=float)
    def neg_ll(beta):
        z = X @ beta
        return -np.sum(T * z - np.logaddexp(0, z))
    res = minimize(neg_ll, np.zeros(X.shape[1]), method="BFGS")
    return 1 / (1 + np.exp(-(X @ res.x)))


def nn_match_att(X, T, Y, caliper: float = None) -> dict:
    """1:1 nearest-neighbor matching on the estimated propensity, with replacement."""
    X = np.asarray(X, dtype=float); T = np.asarray(T, dtype=int); Y = np.asarray(Y, dtype=float)
    e_hat = estimate_propensity(X, T)
    idx_t = np.where(T == 1)[0]; idx_c = np.where(T == 0)[0]
    pair_diff = []
    matches = []
    for i in idx_t:
        d = np.abs(e_hat[i] - e_hat[idx_c])
        j_local = int(np.argmin(d))
        j = int(idx_c[j_local])
        if caliper is not None and d[j_local] > caliper: continue
        pair_diff.append(Y[i] - Y[j])
        matches.append((int(i), j))
    att = float(np.mean(pair_diff))
    se = float(np.std(pair_diff, ddof=1) / math.sqrt(len(pair_diff)))
    # Balance diagnostics
    smd = {}
    for k in range(X.shape[1]):
        tr = X[idx_t, k]; cr = X[[m[1] for m in matches], k]
        smd[f"x{k}"] = float((tr.mean() - cr.mean()) / math.sqrt((tr.var() + cr.var()) / 2))
    return {"ATT": att, "SE_naive": se,
            "n_treated": int(len(idx_t)), "n_matched_pairs": int(len(pair_diff)),
            "propensity_mean_treated": float(e_hat[idx_t].mean()),
            "propensity_mean_control": float(e_hat[idx_c].mean()),
            "smd_matched": smd,
            "method": "1:1 nearest-neighbor propensity matching (with replacement)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    # Treatment depends on covariates (confounded); treatment effect = 2.0
    logit_e = -0.5 + 1.0 * x1 - 0.5 * x2
    T = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_e))).astype(int)
    Y = 1 + 2 * T + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])

    print(f"=== N = {n}, treated = {int(T.sum())} ===")
    print("\n=== Naive difference in means (biased by confounding) ===")
    print(f"  Y_treat - Y_control = {Y[T == 1].mean() - Y[T == 0].mean():.3f}")

    print("\n=== ATT via 1:1 NN propensity matching ===")
    r = nn_match_att(X, T, Y)
    print(f"  ATT (matched)         = {r['ATT']:.3f}  (true 2.0)")
    print(f"  SE (naive)            = {r['SE_naive']:.3f}")
    print(f"  propensity mean treated = {r['propensity_mean_treated']:.3f}")
    print(f"  propensity mean control = {r['propensity_mean_control']:.3f}")
    print(f"  matched-sample SMDs:  {r['smd_matched']}")

    print("\n--- library cross-check (causalinference / DoWhy / R MatchIt) ---")
    print("  R: MatchIt::matchit(T ~ x1 + x2, method = 'nearest', replace = TRUE)")

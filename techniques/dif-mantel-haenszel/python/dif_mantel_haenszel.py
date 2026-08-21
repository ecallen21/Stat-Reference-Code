"""Differential Item Functioning (DIF) via Mantel-Haenszel + logistic
regression (Reference §22.11).

DIF: an item favours or disadvantages a subgroup (gender, ethnicity, ...)
at the SAME ability level.  Distinct from item impact (group mean
differences that are legitimate).

Mantel-Haenszel DIF (Holland-Thayer 1988)
    Stratify examinees by total-score levels.  Within each stratum, form a
    2x2 table (correct/incorrect x reference/focal group).  Combine across
    strata via Mantel-Haenszel odds ratio:
        alpha_MH = (sum_k n_{RC,k} n_{FI,k} / n_k) / (sum_k n_{RI,k} n_{FC,k} / n_k)
    ETS delta:  Delta_MH = -2.35 log(alpha_MH)
        |Delta| < 1.0  negligible (Type A)
        1.0 - 1.5      moderate (Type B)
        > 1.5          large (Type C)

Logistic-regression DIF (Swaminathan-Rogers 1990)
    logit P(y_j = 1) = beta_0 + beta_1 * total + beta_2 * group + beta_3 * total x group
    beta_2 significant -> uniform DIF; beta_3 significant -> non-uniform DIF.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def mh_dif(Y, group, item: int, n_strata: int = 5) -> dict:
    """Mantel-Haenszel DIF test for a single item stratified by total-score-minus-item."""
    Y = np.asarray(Y, dtype=int); group = np.asarray(group)
    n, K = Y.shape
    total_minus = Y.sum(axis=1) - Y[:, item]         # rest score
    # Stratify into quantile bins
    edges = np.quantile(total_minus, np.linspace(0, 1, n_strata + 1))
    strata = np.digitize(total_minus, edges[1:-1])
    num = 0.0; den = 0.0
    for k in np.unique(strata):
        mask = strata == k
        yk = Y[mask, item]; gk = group[mask]
        # ref/focal x correct/incorrect
        R = gk == 0; F = gk == 1
        rc = int((R & (yk == 1)).sum()); ri = int((R & (yk == 0)).sum())
        fc = int((F & (yk == 1)).sum()); fi = int((F & (yk == 0)).sum())
        n_k = rc + ri + fc + fi
        if n_k == 0: continue
        num += rc * fi / n_k; den += ri * fc / n_k
    alpha_MH = num / den if den > 0 else float("nan")
    delta = -2.35 * math.log(alpha_MH) if alpha_MH > 0 else float("nan")
    return {"alpha_MH": float(alpha_MH), "delta_MH": float(delta),
            "flag": ("A negligible" if abs(delta) < 1.0
                     else "B moderate" if abs(delta) < 1.5
                     else "C large"),
            "method": "Mantel-Haenszel DIF (Holland-Thayer)"}


def logistic_dif(Y, group, item: int) -> dict:
    """Swaminathan-Rogers logistic-regression DIF test (uniform + non-uniform)."""
    Y = np.asarray(Y, dtype=float); group = np.asarray(group, dtype=float)
    n, K = Y.shape
    y = Y[:, item]
    total_minus = Y.sum(axis=1) - y
    x = total_minus
    # Model M1: y ~ total; M2: y ~ total + group; M3: y ~ total + group + total*group
    def fit(X):
        def neg_ll(b):
            z = X @ b; return -np.sum(y * z - np.logaddexp(0, z))
        res = minimize(neg_ll, np.zeros(X.shape[1]), method="BFGS")
        return float(-res.fun)
    ll1 = fit(np.column_stack([np.ones(n), x]))
    ll2 = fit(np.column_stack([np.ones(n), x, group]))
    ll3 = fit(np.column_stack([np.ones(n), x, group, x * group]))
    chi_uniform = 2 * (ll2 - ll1); p_uniform = float(stats.chi2.sf(chi_uniform, 1))
    chi_nonuni = 2 * (ll3 - ll2); p_nonuni = float(stats.chi2.sf(chi_nonuni, 1))
    return {"chi2_uniform_DIF": float(chi_uniform), "p_uniform": p_uniform,
            "chi2_nonuniform_DIF": float(chi_nonuni), "p_nonuniform": p_nonuni,
            "method": "Logistic-regression DIF (Swaminathan-Rogers)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, K = 800, 10
    group = rng.choice([0, 1], n)                    # 0 = reference, 1 = focal
    theta = rng.normal(0, 1, n)
    b_item = np.linspace(-2, 2, K)
    b_item_focal = b_item.copy(); b_item_focal[4] += 0.8    # item 4 harder for focal
    b_used = np.where(group[:, None] == 1, b_item_focal[None, :], b_item[None, :])
    P = 1 / (1 + np.exp(-(theta[:, None] - b_used)))
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    print("=== Mantel-Haenszel DIF ===")
    for item in (0, 4, 8):
        r = mh_dif(Y, group, item=item)
        marker = "* DIF ITEM" if abs(r["delta_MH"]) > 1.0 else "  ok"
        print(f"  item {item}: alpha_MH = {r['alpha_MH']:.3f}, Delta = {r['delta_MH']:6.3f}  ({r['flag']}) {marker}")

    print("\n=== Logistic-regression DIF (uniform + non-uniform) ===")
    for item in (0, 4, 8):
        r = logistic_dif(Y, group, item=item)
        print(f"  item {item}: chi2 uniform = {r['chi2_uniform_DIF']:.3f} (p = {r['p_uniform']:.4f}),  "
              f"nonuniform = {r['chi2_nonuniform_DIF']:.3f} (p = {r['p_nonuniform']:.4f})")

    print("\n--- library cross-check (R difR::difMH / difLogistic) ---")

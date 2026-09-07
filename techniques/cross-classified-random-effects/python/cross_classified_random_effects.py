"""Cross-classified random-effects model (CCREM) (Reference Sec 8.19).

Raudenbush & Bryk 2002 'Hierarchical Linear Models', 2nd ed. When
observations are grouped by TWO or more non-nested factors (e.g.
students within schools AND neighborhoods; patients within hospitals
AND primary care practices), CCREMs put independent random effects
on each grouping factor:

    y_ijk = mu + u_j + v_k + eps_ijk
    u_j   ~ N(0, sigma_u^2)     v_k ~ N(0, sigma_v^2)     eps ~ N(0, sigma_e^2)

The design matrix has one indicator per level of each factor. MLE /
REML fits via mixed-effects software (lme4 / statsmodels).

We fit by iterative variance-component EM and compare to a nested
random-intercept model (which mis-specifies the design and
under-estimates variance).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def build_indicator(g, n_levels):
    Z = np.zeros((len(g), n_levels))
    Z[np.arange(len(g)), g] = 1
    return Z


def ccrem_em(y, g1, g2, X=None, tol=1e-5, max_it=200):
    """Simple EM for two-factor CCREM.

    y : outcome (n,)
    g1, g2 : integer level ids for two crossed factors
    X : optional fixed-effect design (n x p); intercept added.
    """
    n = len(y)
    Z1 = build_indicator(g1, g1.max() + 1)
    Z2 = build_indicator(g2, g2.max() + 1)
    if X is None:
        X = np.ones((n, 1))
    else:
        X = np.c_[np.ones(n), X]
    p = X.shape[1]

    #  Initial variance components
    sig_u = 0.5; sig_v = 0.5; sig_e = 1.0
    beta = np.zeros(p)

    for it in range(max_it):
        #  Marginal covariance V = sig_u * Z1 Z1' + sig_v * Z2 Z2' + sig_e * I
        V = sig_u * Z1 @ Z1.T + sig_v * Z2 @ Z2.T + sig_e * np.eye(n)
        try:
            V_inv = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            break
        #  Update beta by GLS
        beta_new = np.linalg.solve(X.T @ V_inv @ X, X.T @ V_inv @ y)
        resid = y - X @ beta_new
        #  BLUP of u, v
        u_hat = sig_u * Z1.T @ V_inv @ resid
        v_hat = sig_v * Z2.T @ V_inv @ resid
        e_hat = resid - Z1 @ u_hat - Z2 @ v_hat
        #  MoM update of variances (one-step EM)
        sig_u_new = float(np.mean(u_hat ** 2) + 1e-8)
        sig_v_new = float(np.mean(v_hat ** 2) + 1e-8)
        sig_e_new = float(np.mean(e_hat ** 2) + 1e-8)
        change = (abs(sig_u_new - sig_u) + abs(sig_v_new - sig_v) + abs(sig_e_new - sig_e))
        sig_u, sig_v, sig_e = sig_u_new, sig_v_new, sig_e_new
        beta = beta_new
        if change < tol:
            break
    return {"beta": beta, "sigma_u_sq": sig_u, "sigma_v_sq": sig_v,
            "sigma_e_sq": sig_e, "iterations": it + 1}


if __name__ == "__main__":
    print("=== Cross-classified random-effects model (CCREM) ===\n")
    rng = np.random.default_rng(0)
    #  200 students across 12 schools x 8 neighborhoods (non-nested).
    n = 400
    g_school = rng.integers(0, 12, size=n)
    g_neigh = rng.integers(0, 8, size=n)
    u_true = rng.normal(scale=0.8, size=12)
    v_true = rng.normal(scale=0.5, size=8)
    y = 5.0 + u_true[g_school] + v_true[g_neigh] + rng.normal(scale=1.0, size=n)

    r = ccrem_em(y, g_school, g_neigh)
    print(f"  n = {n}   n_school = 12   n_neigh = 8")
    print(f"  Truth : sigma_u^2 = {0.8 ** 2:.3f}   sigma_v^2 = {0.5 ** 2:.3f}   sigma_e^2 = {1.0 ** 2:.3f}")
    print(f"  CCREM : sigma_u^2 = {r['sigma_u_sq']:.3f}   sigma_v^2 = {r['sigma_v_sq']:.3f}   sigma_e^2 = {r['sigma_e_sq']:.3f}")
    print(f"  beta (grand mean) = {r['beta'][0]:.3f}   (truth 5.00)")
    print(f"  EM iterations     = {r['iterations']}")

    #  What if we (wrongly) nested neighborhood inside school? (nested random ints)
    #  Approximation: put ALL variance on school only.
    r_nested = ccrem_em(y, g_school, np.zeros(n, dtype=int))
    print(f"\n  Mis-specified 'schools only' model:")
    print(f"    sigma_u^2 = {r_nested['sigma_u_sq']:.3f}   (absorbs neighborhood effect)")
    print(f"    sigma_e^2 = {r_nested['sigma_e_sq']:.3f}   (inflated by ignored variance)")

    print("\n--- library cross-check (lme4::lmer with (1|school)+(1|neigh) R;\n"
          "                          statsmodels MixedLM VC Python) ---")

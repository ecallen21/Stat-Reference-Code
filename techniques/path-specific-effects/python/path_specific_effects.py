"""Path-specific effects (Reference Sec 15.41).

Avin, Shpitser & Pearl 2005; VanderWeele & Chiba 2014. Extends natural
direct/indirect effects to graphs with MULTIPLE mediators, decomposing
the total effect along specific causal PATHWAYS in a DAG.

Setup with two ordered mediators M1, M2:
    T -> M1 -> M2 -> Y
    T -> M1 -> Y            (M1 direct)
    T -> M2 -> Y            (M2 direct, given T's own effect on M2)
    T -> Y                  (all-direct)

Decomposition of TE = E[Y(1) - Y(0)]:

    TE  = PSE_direct + PSE_via_M1 + PSE_via_M2
        = ND-effect + effect through M1-only + effect through M2-only

For linear-Gaussian SCMs (Wright's path coefficients):

    Path T -> Y direct                = beta_1
    Path T -> M1 -> Y                 = alpha_1 * gamma_1     (M1-only)
    Path T -> M2 -> Y                 = alpha_2 * gamma_2     (M2-only, ignoring M1)
    Path T -> M1 -> M2 -> Y           = alpha_1 * mu_1 * gamma_2  (through M1 -> M2)

We fit the four regressions:
    M1 = alpha_0 + alpha_1 * T + eps_M1
    M2 = mu_0 + mu_1 * M1 + alpha_2 * T + eps_M2
    Y  = beta_0 + beta_1 * T + gamma_1 * M1 + gamma_2 * M2 + eps_Y

then compute path products.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def linear_path_effects(t, m1, m2, y):
    n = len(t)
    a = ols(np.c_[np.ones(n), t], m1)
    m = ols(np.c_[np.ones(n), m1, t], m2)
    b = ols(np.c_[np.ones(n), t, m1, m2], y)
    alpha0, alpha1 = a[0], a[1]
    mu0, mu1, alpha2 = m[0], m[1], m[2]
    beta0, beta1, gamma1, gamma2 = b[0], b[1], b[2], b[3]

    path_direct     = beta1                        # T -> Y
    path_via_M1     = alpha1 * gamma1              # T -> M1 -> Y
    path_via_M2     = alpha2 * gamma2              # T -> M2 -> Y (independent of M1)
    path_via_M1_M2  = alpha1 * mu1 * gamma2        # T -> M1 -> M2 -> Y

    total = path_direct + path_via_M1 + path_via_M2 + path_via_M1_M2
    return {"direct": path_direct, "via_M1_only": path_via_M1,
            "via_M2_only": path_via_M2, "via_M1_M2_chain": path_via_M1_M2,
            "total": total, "coefs": {"alpha1": alpha1, "alpha2": alpha2,
                                       "mu1": mu1, "gamma1": gamma1,
                                       "gamma2": gamma2, "beta1": beta1}}


if __name__ == "__main__":
    print("=== Path-specific effects (linear SCM) ===\n")
    rng = np.random.default_rng(0)
    n = 6000

    #  True structural coefficients:
    #    alpha1 = 0.5 (T -> M1)
    #    alpha2 = 0.3 (T -> M2)
    #    mu1    = 0.6 (M1 -> M2)
    #    beta1  = 0.4 (T -> Y direct)
    #    gamma1 = 0.7 (M1 -> Y)
    #    gamma2 = 0.5 (M2 -> Y)
    t = rng.integers(0, 2, size=n).astype(float)
    m1 = 0.5 * t + rng.normal(scale=0.5, size=n)
    m2 = 0.6 * m1 + 0.3 * t + rng.normal(scale=0.5, size=n)
    y = 0.4 * t + 0.7 * m1 + 0.5 * m2 + rng.normal(scale=1.0, size=n)

    r = linear_path_effects(t, m1, m2, y)
    truths = {"direct": 0.4, "via_M1_only": 0.5 * 0.7,
              "via_M2_only": 0.3 * 0.5, "via_M1_M2_chain": 0.5 * 0.6 * 0.5}
    truths["total"] = sum(truths.values())

    print(f"  {'path':>20s}  {'estimate':>10s}  {'truth':>10s}")
    for k in ["direct", "via_M1_only", "via_M2_only", "via_M1_M2_chain", "total"]:
        print(f"  {k:>20s}  {r[k]:>+10.4f}  {truths[k]:>+10.4f}")

    print(f"\n  Fitted coefs = {r['coefs']}")

    print("\n--- library cross-check (paths R, medflex R; DoWhy Python) ---")

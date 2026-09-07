"""Principal stratification -- CACE / LATE (Reference Sec 15.32).

Frangakis & Rubin 2002; Angrist, Imbens & Rubin 1996. When treatment
assignment Z is randomised but received treatment D is not (imperfect
compliance), the average treatment effect in the whole trial is diluted.
The Complier Average Causal Effect (CACE, a.k.a. LATE) estimates the
effect among compliers -- units who would take the treatment iff assigned.

Principal strata (for binary Z, D):

    always-taker  A: D(1) = D(0) = 1
    complier      C: D(1) = 1, D(0) = 0
    defier        F: D(1) = 0, D(0) = 1
    never-taker   N: D(1) = D(0) = 0

Under randomisation + exclusion restriction + monotonicity (no defiers):

    ITT       = E[Y | Z=1] - E[Y | Z=0]        (intent-to-treat)
    ITT_D     = E[D | Z=1] - E[D | Z=0]        (share of compliers)
    CACE / IV = ITT / ITT_D                    (Wald / 2SLS estimator)

Equivalently: CACE = Cov(Y, Z) / Cov(D, Z) -- the 2SLS estimator with Z
as instrument for D. When Z is randomised the 4 assumptions reduce to:
  * Randomisation of Z
  * Exclusion restriction: Z affects Y only through D
  * Monotonicity: no defiers
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def cace_wald(y, d, z):
    """Wald / IV estimator for CACE."""
    itt = np.mean(y[z == 1]) - np.mean(y[z == 0])
    itt_d = np.mean(d[z == 1]) - np.mean(d[z == 0])
    return {"ITT": itt, "ITT_D": itt_d, "CACE": itt / itt_d}


def cace_2sls(y, d, z, x=None):
    """Two-stage least squares -- equivalent to Wald when no covariates."""
    n = len(y)
    if x is None:
        x = np.zeros((n, 0))
    #  Stage 1: regress d on [1, x, z] -> d_hat
    A1 = np.c_[np.ones(n), x, z]
    beta1 = np.linalg.lstsq(A1, d, rcond=None)[0]
    d_hat = A1 @ beta1
    #  Stage 2: regress y on [1, x, d_hat]
    A2 = np.c_[np.ones(n), x, d_hat]
    beta2 = np.linalg.lstsq(A2, y, rcond=None)[0]
    return {"CACE": beta2[-1]}


def compliance_shares(d, z):
    """Point-identify strata shares under monotonicity + randomisation."""
    p_at = np.mean(d[z == 0])                  # always-takers (D=1 when Z=0)
    p_nt = 1 - np.mean(d[z == 1])              # never-takers  (D=0 when Z=1)
    p_c = 1 - p_at - p_nt                     # compliers by subtraction
    return {"always_taker": p_at, "never_taker": p_nt, "complier": p_c}


if __name__ == "__main__":
    print("=== Principal stratification -- CACE via Wald IV ===\n")
    rng = np.random.default_rng(0)
    n = 8000

    #  40% compliers, 20% always-takers, 40% never-takers
    strata = rng.choice(["C", "A", "N"], size=n, p=[0.4, 0.2, 0.4])
    z = rng.integers(0, 2, size=n)
    d = np.where(strata == "A", 1,
        np.where(strata == "N", 0, z))
    #  True CACE = 2.0; always/never-taker effect = 0 (exclusion restriction)
    tau_true = 2.0
    y = 1.0 + 0.5 * (strata == "A") + rng.normal(scale=1, size=n)
    y = y + tau_true * (strata == "C") * z

    shares = compliance_shares(d, z)
    print(f"  True shares:      C=0.40, A=0.20, N=0.40")
    print(f"  Estimated shares: C={shares['complier']:.2f}, "
          f"A={shares['always_taker']:.2f}, N={shares['never_taker']:.2f}")

    r = cace_wald(y, d, z)
    print(f"\n  ITT (naive)       = {r['ITT']:+.3f}")
    print(f"  ITT_D (compliance)= {r['ITT_D']:+.3f}")
    print(f"  CACE (Wald / IV)  = {r['CACE']:+.3f}   (truth {tau_true:+.2f})")

    r2 = cace_2sls(y, d, z)
    print(f"  CACE (2SLS)       = {r2['CACE']:+.3f}")

    print("\n--- library cross-check (ivreg::ivreg R; linearmodels.IV2SLS Python) ---")

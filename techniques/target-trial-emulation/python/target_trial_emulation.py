"""Target trial emulation (Reference Sec 43.11).

Hernan & Robins (2016).  For a comparative-effectiveness question,
first WRITE THE PROTOCOL of the ideal randomised trial (eligibility,
treatment strategies, assignment, follow-up, outcome, causal contrast,
analysis) that you would run if resources allowed.  Then EMULATE
each element from claims / EHR data.

Compact demo:
  * Eligibility: adults with disease X, no contra-indications, no
    prior use of either study drug.
  * Treatment strategies: drug A vs drug B (new-user, active-
    comparator).
  * Time zero: first prescription day.
  * Follow-up: from time zero to outcome, discontinuation, or
    censoring.
  * Analysis: IPTW hazard-ratio estimate.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression


def emulate(cohort, treatment_var, outcome_var, covariate_cols):
    """Compute IPTW ATE from an emulated new-user, active-comparator design."""
    X = cohort[:, covariate_cols].astype(float)
    T = cohort[:, treatment_var].astype(int)
    y = cohort[:, outcome_var].astype(float)
    m = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500).fit(X, T)
    ps = m.predict_proba(X)[:, 1].clip(0.05, 0.95)
    w = np.where(T == 1, 1 / ps, 1 / (1 - ps))
    ate = float(np.average(y[T == 1], weights=w[T == 1])
                - np.average(y[T == 0], weights=w[T == 0]))
    return {"ATE_IPTW": ate, "n_treated": int(T.sum()), "n_control": int((1 - T).sum())}


if __name__ == "__main__":
    print("=== Target trial emulation on claims-like data ===\n")
    rng = np.random.default_rng(0)
    n = 3000
    # Covariates: age (0-1 scaled), comorbidity index (0-1)
    age = rng.uniform(0, 1, n)
    comorb = rng.uniform(0, 1, n)
    # Treatment (drug A=1 vs drug B=0) depends on covariates -> confounding
    logit_T = -0.5 + 1.0 * age + 0.8 * comorb + rng.normal(0, 0.5, n)
    T = (rng.random(n) < 1 / (1 + np.exp(-logit_T))).astype(int)
    # True ATE of A vs B on continuous outcome = 0.3
    y = 2.0 + 0.3 * T + 1.0 * age + 0.5 * comorb + rng.normal(0, 0.5, n)

    # Assemble columns: [age, comorb, T, y]
    cohort = np.column_stack([age, comorb, T, y])
    naive = float(y[T == 1].mean() - y[T == 0].mean())
    r = emulate(cohort, treatment_var=2, outcome_var=3, covariate_cols=[0, 1])

    print(f"  True ATE = 0.30")
    print(f"  Naive difference          = {naive:.3f}   (confounded)")
    print(f"  Target-trial-emulated ATE = {r['ATE_IPTW']:.3f}   (IPTW)")
    print(f"  n treated = {r['n_treated']}, n control = {r['n_control']}\n")

    print("  Key protocol elements to emulate (Hernan-Robins 2016):")
    print("    1. Eligibility criteria applied at time zero.")
    print("    2. Treatment strategies (new-user, active-comparator).")
    print("    3. Assignment (mimicked via propensity or grace-period design).")
    print("    4. Follow-up start and end.")
    print("    5. Outcome definition + measurement.")
    print("    6. Causal contrast (per-protocol vs intention-to-treat).")
    print("    7. Statistical analysis (IPTW / g-methods).\n")
    print("--- library cross-check (R TrialEmulation, CohortMethod OHDSI; Python zepid) ---")

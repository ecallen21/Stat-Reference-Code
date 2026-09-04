"""Confounding by indication, channelling, protopathic bias (Reference Sec 43.12).

CONFOUNDING BY INDICATION: sicker patients get treated -> apparent
  treatment-outcome association reflects underlying disease severity.
CHANNELING BIAS: prescribers systematically prescribe certain drugs
  to certain risk groups.
PROTOPATHIC BIAS: symptoms of an as-yet-undiagnosed disease trigger
  a prescription that is then mistakenly implicated as the cause.

Fixes:
  * New-user, active-comparator design (see companion technique).
  * Adjust for INDICATION severity (biomarker, disease stage).
  * Instrumental-variable methods where feasible.
"""
from __future__ import annotations    # stdlib

import warnings
warnings.filterwarnings("ignore")

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    print("=== Confounding by indication: naive vs adjusted vs active-comparator ===\n")
    rng = np.random.default_rng(0)
    n = 4000
    severity = rng.normal(0, 1, n)                # unmeasured underlying disease severity
    # Probability of getting drug A (vs no drug) rises steeply with severity
    p_treat = 1 / (1 + np.exp(-(1.5 * severity)))
    T = (rng.random(n) < p_treat).astype(int)
    # Outcome (mortality) driven by severity + weak protective effect of drug (true log HR = -0.3)
    # Simulated as continuous risk
    y = 1.0 * severity - 0.3 * T + rng.normal(0, 0.5, n)

    # NAIVE analysis: apparent effect
    naive = float(y[T == 1].mean() - y[T == 0].mean())
    print(f"  Naive mean difference (T=1 - T=0)         = {naive:+.3f}   (biased; should be -0.3)")

    # ADJUSTED for severity (if measured)
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(n), T, severity])
    beta, *_ = lstsq(X, y, rcond=None)
    print(f"  Regression coefficient on T adj. for severity = {beta[1]:+.3f}   (unbiased)")

    # ACTIVE COMPARATOR proxy: compare drug A users with drug B users (both indicated for the same condition)
    # Simulate drug B group among the severe subset:
    treatB = (severity > 0.5) & (rng.random(n) < 0.35)
    tA = T & ~treatB
    # (drug A vs drug B, both indicated so severity is more similar)
    diff_active = float(y[tA == 1].mean() - y[treatB == 1].mean())
    print(f"  Active-comparator (A vs B, both indicated)  = {diff_active:+.3f}   (bias reduced)")
    print()

    print("  Protopathic bias caveat: if the outcome symptoms triggered the prescription,")
    print("  even a within-person design will misattribute causation.  Lag the exposure")
    print("  window by weeks-to-months to attenuate this bias.\n")
    print("--- library cross-check (R MatchIt/WeightIt/CohortMethod; Python causalinference/zepid/dowhy) ---")

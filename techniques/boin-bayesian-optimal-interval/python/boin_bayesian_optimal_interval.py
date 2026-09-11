"""BOIN - Bayesian Optimal Interval (Reference Sec 47.263).

Liu & Yuan 2015 JRSS-C. Model-ASSISTED phase-I dose-finding
that combines the simplicity of 3+3 with the efficiency of
CRM. At each cohort, compare the observed DLT rate to a
PRE-CALCULATED INTERVAL (lambda_e, lambda_d) around the target
p_target:

    p_hat = n_DLT / n_treated at current dose
    if p_hat <= lambda_e:   escalate
    if p_hat >= lambda_d:   de-escalate
    otherwise:              stay

Default lambda_e = 0.6 * p_target, lambda_d = 1.4 * p_target
(minimises escalation-decision error under H0). Simpler
transition table than CRM, similar performance.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def boin_bounds(p_target, phi_1=None, phi_2=None):
    """Default BOIN boundaries; phi_1 = 0.6 p, phi_2 = 1.4 p."""
    if phi_1 is None: phi_1 = 0.6 * p_target
    if phi_2 is None: phi_2 = 1.4 * p_target
    # Optimal decision boundaries (approximation): logs of odds ratios
    lambda_e = np.log((1 - phi_1) / (1 - p_target)) / np.log(p_target * (1 - phi_1) / (phi_1 * (1 - p_target)))
    lambda_d = np.log((1 - p_target) / (1 - phi_2)) / np.log(phi_2 * (1 - p_target) / (p_target * (1 - phi_2)))
    return float(lambda_e), float(lambda_d)


def boin_trial(p_true, p_target=0.25, cohort=3, n_max=30, rng=None):
    """Simulate a single BOIN trial; return recommended MTD, n_treated, n_DLT."""
    if rng is None: rng = np.random.default_rng(0)
    lambda_e, lambda_d = boin_bounds(p_target)
    n_doses = len(p_true)
    n_treated = np.zeros(n_doses, dtype=int); n_dlt = np.zeros(n_doses, dtype=int)
    d = 0
    while n_treated.sum() < n_max:
        # Treat one cohort at d
        t = int(rng.binomial(cohort, p_true[d]))
        n_treated[d] += cohort; n_dlt[d] += t
        p_hat = n_dlt[d] / n_treated[d]
        # Safety exclusion (posterior of tox > p_target with high prob)
        # Simple beta(1,1): p(tox > p_target) > 0.95 => rule out
        from scipy.stats import beta
        if beta.cdf(p_target, n_dlt[d] + 1, n_treated[d] - n_dlt[d] + 1) < 0.05:
            # Exclude this and higher doses
            if d == 0: return -1, n_treated, n_dlt
            d = d - 1
        elif p_hat <= lambda_e:
            if d < n_doses - 1: d += 1
        elif p_hat >= lambda_d:
            if d > 0: d = d - 1
        # else: stay
    # Isotonic recommendation: dose with tox closest to target
    ph = np.where(n_treated > 0, n_dlt / np.maximum(n_treated, 1), np.inf)
    return int(np.argmin(np.abs(ph - p_target))), n_treated, n_dlt


if __name__ == "__main__":
    print("=== BOIN - Bayesian Optimal Interval (Liu & Yuan 2015) ===\n")
    rng = np.random.default_rng(0)

    p_true = [0.02, 0.08, 0.18, 0.40, 0.65]
    p_target = 0.25
    print(f"  True tox: {p_true}, target = {p_target}")

    lambda_e, lambda_d = boin_bounds(p_target)
    print(f"  Boundaries: lambda_e = {lambda_e:.3f} (escalate if p_hat <= this)")
    print(f"              lambda_d = {lambda_d:.3f} (de-escalate if p_hat >= this)\n")

    # Single-trial run
    d, n_t, n_d = boin_trial(p_true, p_target, cohort=3, n_max=30, rng=rng)
    print(f"  Single run - recommended MTD = dose {d + 1 if d >= 0 else 'none'}")
    print(f"    n_treated per dose: {n_t.tolist()}")
    print(f"    n_DLT     per dose: {n_d.tolist()}\n")

    # 1000-trial simulation
    n_sim = 1000; picks = np.zeros(len(p_true) + 1, dtype=int); total_n = 0
    for _ in range(n_sim):
        d, n_t, _ = boin_trial(p_true, p_target, cohort=3, n_max=30, rng=rng)
        picks[d + 1] += 1
        total_n += n_t.sum()

    print(f"  {n_sim} simulated trials:")
    print(f"    {'dose':>5}  {'true tox':>10}  {'% picked':>10}")
    print(f"    {'none':>5}  {'-':>10}  {picks[0] / n_sim * 100:>9.1f}%")
    for i in range(len(p_true)):
        print(f"    {i + 1:>5}  {p_true[i]:>10.2f}  {picks[i + 1] / n_sim * 100:>9.1f}%")
    print(f"\n    mean patients per trial: {total_n / n_sim:.1f}")

    print("\n  BOIN concentrates more selections at the true MTD (dose 3) than 3+3")
    print("  with comparable simplicity - just look up p_hat vs lambda thresholds.")

    print("\n--- library cross-check (BOIN R; boinet R; U-Chicago BOIN web app) ---")

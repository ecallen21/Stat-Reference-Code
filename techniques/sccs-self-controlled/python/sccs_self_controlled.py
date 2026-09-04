"""Self-controlled case series (Reference Sec 43.2).

Farrington 1995.  ONLY cases (people with the outcome) are analysed.
Each case contributes their observation time split into RISK periods
(shortly after exposure) and BASELINE periods.  Under a conditional
Poisson likelihood the incidence rate ratio (IRR) of risk vs
baseline is estimated within each person, so all TIME-INVARIANT
confounders (genetics, chronic disease, socio-economic) drop out.

Model:  N_ij ~ Poisson(mu_ij),  log(mu_ij) = phi_i + log(e_ij) + beta * x_ij
  phi_i = individual intercept (nuisance), removed by conditioning.
  e_ij  = time exposed in interval j
  x_ij  = 1 if interval j is a risk window else 0
  beta  = log IRR (parameter of interest)

Conditional-Poisson MLE reduces to an EQUIVALENT LOGISTIC-LIKE score
over cases.  Here we implement the compact 2-window (risk vs baseline)
case: for each case i, let e_i^R and e_i^B be person-time in risk /
baseline periods and n_i^R, n_i^B be events in each; sum events and
times, then IRR = (sum n^R / sum e^R) / (sum n^B / sum e^B), with SE
from the delta method on log IRR.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sccs_2window(risk_events, risk_time, base_events, base_time):
    """Conditional-Poisson IRR estimate + Wald CI (2-window SCCS)."""
    n_R = risk_events.sum(); e_R = risk_time.sum()
    n_B = base_events.sum(); e_B = base_time.sum()
    irr = (n_R / e_R) / (n_B / e_B)
    log_irr = np.log(irr)
    se = np.sqrt(1 / n_R + 1 / n_B)
    return {"IRR": float(irr), "log_IRR": float(log_irr),
            "SE_log_IRR": float(se),
            "CI95": (float(np.exp(log_irr - 1.96 * se)),
                     float(np.exp(log_irr + 1.96 * se)))}


if __name__ == "__main__":
    print("=== Self-controlled case series: 2-window IRR ===\n")
    rng = np.random.default_rng(0)
    n_cases = 500
    # Baseline rate 0.5 events / person-year; risk-window rate 1.5 (IRR = 3)
    base_time = rng.uniform(0.5, 5, n_cases)              # years
    risk_time = np.full(n_cases, 0.1)                      # 5-week window
    lam_B = 0.5; lam_R = 1.5
    n_B = rng.poisson(lam_B * base_time)
    n_R = rng.poisson(lam_R * risk_time)

    r = sccs_2window(n_R, risk_time, n_B, base_time)
    print(f"  True IRR = 3.0")
    print(f"  Estimated IRR = {r['IRR']:.3f}   log IRR = {r['log_IRR']:.3f}"
          f"   SE = {r['SE_log_IRR']:.3f}")
    print(f"  95% CI: ({r['CI95'][0]:.3f}, {r['CI95'][1]:.3f})\n")

    print("--- library cross-check (R SCCS::standardsccs, SelfControlledCaseSeries; Python custom + rpy2) ---")

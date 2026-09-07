"""Mediation -- natural direct/indirect effects (Reference Sec 15.38).

VanderWeele 2015 'Explanation in Causal Inference: Methods for
Mediation and Interaction', OUP. Extends the classical Baron & Kenny
mediation decomposition to potential-outcomes causal inference,
handling exposure-mediator interaction correctly.

Setup: exposure T (0/1), mediator M, outcome Y, covariates C.

  Y(t, m)  = counterfactual outcome under exposure t, mediator m
  M(t)     = counterfactual mediator under exposure t

Decomposition of the total effect (TE = E[Y(1, M(1))] - E[Y(0, M(0))]):

  NDE = E[Y(1, M(0))] - E[Y(0, M(0))]   (natural direct)
  NIE = E[Y(1, M(1))] - E[Y(1, M(0))]   (natural indirect)
  TE  = NDE + NIE

Under sequential ignorability (Imai et al. 2010) + no exposure-induced
confounder of M -> Y, NDE and NIE are point-identified. VanderWeele
(2014) closed-form estimator for continuous M, continuous Y, with
mediator model M = alpha0 + alpha1*T + alpha2*C + eps_M and outcome
Y = beta0 + beta1*T + beta2*M + beta3*T*M + beta4*C + eps_Y:

  NDE = (beta1 + beta3 * (alpha0 + alpha2*c_mean)) * 1
  NIE = (beta2 + beta3) * alpha1 * 1
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def vanderweele_nde_nie(y, t, m, c):
    """Closed-form NDE/NIE under continuous M/Y with T-M interaction.

    Returns dict with NDE, NIE, TE at c = c_mean.
    """
    n = len(y)
    C = c.reshape(n, -1)
    #  Mediator model: M ~ 1 + T + C
    Xm = np.c_[np.ones(n), t, C]
    alpha = fit_ols(Xm, m)     # alpha = [a0, a1, a2...]
    a0, a1 = alpha[0], alpha[1]
    a2 = alpha[2:]
    #  Outcome model: Y ~ 1 + T + M + T*M + C
    Xy = np.c_[np.ones(n), t, m, t * m, C]
    beta = fit_ols(Xy, y)      # beta = [b0, b1, b2, b3, b4...]
    b1, b2, b3 = beta[1], beta[2], beta[3]

    c_mean = C.mean(axis=0)
    m_at_control = a0 + c_mean @ a2   # E[M(0) | C=c_mean]
    NDE = b1 + b3 * m_at_control      # change t 0->1, hold M at M(0)
    NIE = (b2 + b3) * a1              # M shifts under t=1
    return {"NDE": NDE, "NIE": NIE, "TE": NDE + NIE,
            "alpha": alpha.tolist(), "beta": beta.tolist()}


if __name__ == "__main__":
    print("=== Mediation -- natural direct/indirect effects (VanderWeele) ===\n")
    rng = np.random.default_rng(0)
    n = 5000
    c = rng.normal(size=n)
    t = rng.integers(0, 2, size=n)
    #  True mediator: M = 0.5 + 0.8*T + 0.3*C + noise
    m = 0.5 + 0.8 * t + 0.3 * c + rng.normal(scale=0.5, size=n)
    #  True outcome: Y = 1.0 + 0.4*T + 0.6*M + 0.2*T*M + 0.1*C + noise
    y = 1.0 + 0.4 * t + 0.6 * m + 0.2 * t * m + 0.1 * c + rng.normal(scale=1.0, size=n)

    #  True effects at c=E[C]=0:
    #    E[M(0)|C=0] = 0.5
    #    NDE = 0.4 + 0.2 * 0.5 = 0.50
    #    NIE = (0.6 + 0.2) * 0.8 = 0.64
    #    TE  = 1.14
    r = vanderweele_nde_nie(y, t, m, c)
    print(f"  Mediator model alpha = {[f'{x:+.3f}' for x in r['alpha']]}")
    print(f"  Outcome  model beta  = {[f'{x:+.3f}' for x in r['beta']]}")
    print()
    print(f"  NDE = {r['NDE']:+.3f}   (truth +0.500)")
    print(f"  NIE = {r['NIE']:+.3f}   (truth +0.640)")
    print(f"  TE  = {r['TE']:+.3f}   (truth +1.140)")
    print(f"  Proportion mediated = {r['NIE'] / r['TE']:.2%}")

    print("\n--- library cross-check (mediation R, medflex R; DoWhy Python) ---")

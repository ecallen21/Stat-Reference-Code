"""Semi-Markov multi-state model (Reference Sec 47.47).

Foucher et al 2010 'A semi-Markov model with covariates for the study
of the AIDS epidemic', LIDA 16. Extends Markov multi-state models
by letting the transition intensity depend on TIME SINCE ENTRY to
the current state (sojourn time), not just calendar time:

    q_{ij}(t | u) = q_{ij}(u)      where u = t - entry into i

We use a parametric Weibull sojourn distribution for each origin
state and a multinomial transition matrix conditional on leaving.
Fit by direct log-likelihood maximisation on interval-observed
transitions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # MLE
from scipy.special import gammaln    # Weibull log-lik constants


def weibull_logpdf(u, shape, scale):
    return (np.log(shape) - np.log(scale) + (shape - 1) * (np.log(u) - np.log(scale))
            - (u / scale) ** shape)


def weibull_logsurv(u, shape, scale):
    return -((u / scale) ** shape)


def semi_markov_loglik(params, sojourns, dest, censor, n_next):
    """Two-state-transient + one absorbing state demo. Params: (shape, scale, p12)."""
    shape, scale, logit_p = params
    if shape <= 0 or scale <= 0:
        return 1e12
    p12 = 1 / (1 + np.exp(-logit_p))
    ll = 0.0
    for u, d, c in zip(sojourns, dest, censor):
        if c:
            ll += weibull_logsurv(u, shape, scale)
        else:
            ll += weibull_logpdf(u, shape, scale)
            ll += np.log(p12) if d == 2 else np.log(1 - p12)
    return -ll


def fit_semi_markov(sojourns, dest, censor):
    x0 = np.array([1.0, np.mean(sojourns), 0.0])
    res = minimize(semi_markov_loglik, x0,
                    args=(sojourns, dest, censor, None),
                    method="Nelder-Mead",
                    options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 20000})
    shape, scale, logit_p = res.x
    return {"shape": float(shape), "scale": float(scale),
            "p_1to2": float(1 / (1 + np.exp(-logit_p))),
            "loglik": -float(res.fun)}


if __name__ == "__main__":
    print("=== Semi-Markov multi-state model (Foucher et al 2010) ===\n")
    rng = np.random.default_rng(0)
    n = 800
    # True: Weibull(shape=1.5, scale=3.0), P(go to state 2 | leave state 1) = 0.7
    true_shape, true_scale, true_p12 = 1.5, 3.0, 0.7
    sojourns = rng.weibull(true_shape, size=n) * true_scale
    dest = rng.binomial(1, true_p12, size=n) + 1    # 1 -> exit to 2, 0 -> exit to 3
    admin_cens = 5.0
    censor = sojourns > admin_cens
    sojourns = np.minimum(sojourns, admin_cens)
    print(f"  n={n}, censor rate = {censor.mean():.3f}")
    print(f"  Truth: Weibull(shape={true_shape}, scale={true_scale}), P(1->2)={true_p12}")

    fit = fit_semi_markov(sojourns, dest, censor)
    print(f"\n  MLE: shape = {fit['shape']:.3f}   scale = {fit['scale']:.3f}   "
          f"P(1->2) = {fit['p_1to2']:.3f}")
    print(f"  Log-lik = {fit['loglik']:.2f}")

    print("\n--- library cross-check (SemiMarkov / mstate R; not standard Python) ---")

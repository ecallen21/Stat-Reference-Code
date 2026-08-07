"""Mixture cure models for long-term survivors (Reference §11.22).

Some survival populations contain a fraction pi of subjects who will NEVER
experience the event (long-term survivors, cured).  Standard survival models
force S(infinity) = 0, which biases estimates when a plateau appears in the
KM curve.

Mixture cure model (Berkson-Gage 1952; Farewell 1982):
    Population survival: S_pop(t) = pi + (1 - pi) * S_u(t)
        pi        : cure probability
        S_u(t)    : survival of the UNCURED subgroup (proper distribution)

Extended (regression) version:
    logit pi(x) = alpha_0 + alpha^T x       (incidence submodel)
    S_u(t | z)  = Weibull, Cox, ...          (latency submodel)

EM estimation
    E-step: compute w_i = Pr(uncured | y_i, censoring)
    M-step: logistic regression for pi with weights w; weighted survival
    MLE for the latency.

The demo below fits a Weibull mixture cure by direct MLE (BFGS on the
mixture likelihood), which is simpler than the EM for small examples.

Contrast with Kaplan-Meier: KM shows the plateau but cannot separately
estimate pi and S_u.  Cure models decompose these.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _weibull_S(t, shape, scale):
    return np.exp(-(t / scale) ** shape)


def _weibull_pdf(t, shape, scale):
    return (shape / scale) * (t / scale) ** (shape - 1) * _weibull_S(t, shape, scale)


def cure_weibull(time, event, X_cure=None) -> dict:
    """Mixture cure model with logistic cure probability + Weibull latency.

    X_cure : covariates for logit(pi); default = intercept only.
    """
    time = np.asarray(time, dtype=float); event = np.asarray(event, dtype=int)
    n = len(time)
    if X_cure is None:
        Xc = np.ones((n, 1))
    else:
        Xc = np.column_stack([np.ones(n), np.asarray(X_cure, dtype=float)])
    q = Xc.shape[1]

    def neg_ll(theta):
        alpha = theta[:q]
        log_shape, log_scale = theta[q], theta[q + 1]
        shape = math.exp(log_shape); scale = math.exp(log_scale)
        pi = 1 / (1 + np.exp(-(Xc @ alpha)))          # cure prob
        S_u = _weibull_S(time, shape, scale)
        f_u = _weibull_pdf(time, shape, scale)
        # Event: p(uncured) * f_u(t)     Censored: p(cured) + p(uncured) * S_u(t)
        loglik_e = np.log((1 - pi) * f_u + 1e-300)
        loglik_c = np.log(pi + (1 - pi) * S_u + 1e-300)
        return -np.sum(np.where(event == 1, loglik_e, loglik_c))
    theta0 = np.concatenate([np.full(q, 0.0), [0.0, math.log(np.median(time[time > 0]) if (time > 0).any() else 1.0)]])
    res = minimize(neg_ll, theta0, method="BFGS")
    alpha = res.x[:q]
    shape = math.exp(res.x[q]); scale = math.exp(res.x[q + 1])
    se = np.sqrt(np.diag(res.hess_inv))
    return {"alpha_cure": alpha,
            "se_alpha_cure": se[:q],
            "weibull_shape": float(shape),
            "weibull_scale": float(scale),
            "cure_prob_baseline": float(1 / (1 + math.exp(-alpha[0]))),
            "log_likelihood": float(-res.fun),
            "n": int(n), "n_events": int(event.sum()),
            "method": "Weibull mixture cure model (MLE)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    pi_true = 0.35  # 35% cured
    is_uncured = rng.uniform(size=n) >= pi_true
    T = np.full(n, np.inf)
    T[is_uncured] = rng.weibull(1.4, size=is_uncured.sum()) * 3.0  # Weibull time-to-event
    C = rng.exponential(5, n)
    time = np.minimum(T, C); event = (T <= C).astype(int)

    print(f"=== Weibull mixture cure MLE, n = {n}, events = {int(event.sum())} ===")
    r = cure_weibull(time, event)
    print(f"  cure prob (baseline) = {r['cure_prob_baseline']:.3f}   (true {pi_true})")
    print(f"  Weibull shape         = {r['weibull_shape']:.3f}   (true 1.4)")
    print(f"  Weibull scale         = {r['weibull_scale']:.3f}   (true 3.0)")
    print(f"  log-lik = {r['log_likelihood']:.2f}")

    print("\n=== With a covariate on the cure probability ===")
    x = rng.normal(size=n)
    is_uncured = rng.uniform(size=n) >= (1 / (1 + np.exp(-(-0.5 + 0.7 * x))))
    T = np.full(n, np.inf)
    T[is_uncured] = rng.weibull(1.4, size=is_uncured.sum()) * 3.0
    C = rng.exponential(5, n)
    time = np.minimum(T, C); event = (T <= C).astype(int)
    r = cure_weibull(time, event, X_cure=x)
    print(f"  cure regression: {r['alpha_cure'].round(3)}  (true -0.5, 0.7)")
    print(f"  Weibull shape/scale: {r['weibull_shape']:.2f}, {r['weibull_scale']:.2f}")

    print("\n--- library cross-check (smcure R; lifelines-Python has cure fitters) ---")
    print("  R: smcure::smcure(Surv(time, event) ~ x, cureform = ~x, model = 'ph')")

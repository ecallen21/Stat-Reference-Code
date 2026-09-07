"""Overlap weighting -- ATO estimand (Reference Sec 15.31).

Li, Morgan & Zaslavsky 2018. A propensity-score weighting scheme whose target
population is the region of clinical equipoise -- units for whom treatment
assignment is genuinely uncertain.

For binary treatment T with propensity e(x) = P(T=1|X=x):

    ATE  weight:  w = T/e  + (1-T)/(1-e)     (target = full population)
    ATT  weight:  w = T    + (1-T)*e/(1-e)   (target = treated pop)
    ATO  weight:  w = T*(1-e) + (1-T)*e      (target = overlap region)

Advantages of ATO:
  * Weights are bounded in [0, 1/2] -- no extreme weights that blow up SE.
  * Exact covariate balance on x for LOGISTIC propensity models.
  * Estimand is the average treatment effect in the OVERLAP population --
    the subset where clinicians are truly uncertain which arm to prescribe.

Estimator (Hajek form):
    ATO_hat = sum(w * T * Y) / sum(w * T)  -  sum(w * (1-T) * Y) / sum(w * (1-T))
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize    # logistic PS fit


def logistic_ps(x, t):
    """Fit logistic propensity by MLE, return e_hat."""
    x1 = np.c_[np.ones(len(t)), x]

    def nll(beta):
        eta = x1 @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(p, 1e-8, 1 - 1e-8)
        return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))

    beta0 = np.zeros(x1.shape[1])
    r = minimize(nll, beta0, method="L-BFGS-B")
    eta = x1 @ r.x
    return 1.0 / (1.0 + np.exp(-eta))


def overlap_weights(t, e):
    """ATO weights: T*(1-e) + (1-T)*e."""
    return t * (1 - e) + (1 - t) * e


def iptw_weights(t, e, target="ate"):
    if target == "ate":
        return t / e + (1 - t) / (1 - e)
    if target == "att":
        return t + (1 - t) * e / (1 - e)
    raise ValueError(target)


def hajek_ate(w, t, y):
    """Weighted mean difference, Hajek (normalised) form."""
    m1 = np.sum(w * t * y) / np.sum(w * t)
    m0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
    return m1 - m0


if __name__ == "__main__":
    print("=== Overlap weighting (ATO) vs IPTW (ATE / ATT) ===\n")
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 3))
    #  Strong confounding + one covariate with extreme prop score tails
    eta = 0.0 + 1.5 * x[:, 0] + 1.0 * x[:, 1] - 0.5 * x[:, 2]
    e_true = 1 / (1 + np.exp(-eta))
    t = (rng.uniform(size=n) < e_true).astype(int)
    #  Heterogeneous effect: bigger in overlap region (moderate e)
    tau = 1.0 + 0.5 * x[:, 0]
    y0 = x @ np.array([0.5, -0.2, 0.3]) + rng.normal(scale=1, size=n)
    y = y0 + tau * t

    e_hat = logistic_ps(x, t)

    for label, tar in [("ATE (IPTW)", "ate"), ("ATT (IPTW)", "att")]:
        w = iptw_weights(t, e_hat, target=tar)
        est = hajek_ate(w, t, y)
        print(f"  {label:12s}  est = {est:+.3f}   max weight = {w.max():.1f}")

    w = overlap_weights(t, e_hat)
    est = hajek_ate(w, t, y)
    print(f"  {'ATO':12s}  est = {est:+.3f}   max weight = {w.max():.3f}")

    #  Approx truth for ATO: E[tau * e*(1-e)] / E[e*(1-e)]
    true_ato = np.sum(tau * e_true * (1 - e_true)) / np.sum(e_true * (1 - e_true))
    print(f"\n  True ATO (overlap-weighted E[tau]) = {true_ato:+.3f}")

    print("\n--- balance check on x[:, 0] ---")
    def wstd_diff(x, t, w):
        m1 = np.sum(w * t * x) / np.sum(w * t)
        m0 = np.sum(w * (1 - t) * x) / np.sum(w * (1 - t))
        return (m1 - m0) / np.sqrt(np.var(x))
    print(f"  Unweighted SMD    = {wstd_diff(x[:, 0], t, np.ones(n)):+.3f}")
    print(f"  IPTW-ATE SMD      = {wstd_diff(x[:, 0], t, iptw_weights(t, e_hat, 'ate')):+.3f}")
    print(f"  Overlap-ATO SMD   = {wstd_diff(x[:, 0], t, overlap_weights(t, e_hat)):+.3f}")

    print("\n--- library cross-check (PSweight R; causalml Python) ---")

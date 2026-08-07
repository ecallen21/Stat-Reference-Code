"""Inverse Probability of Treatment Weighting (Reference §15.7).

IPW / IPTW alternative to matching for causal inference under strong
ignorability.  Reweight the observed sample to CREATE a pseudo-population
in which treatment is independent of X.

Weights (per subject i)
    w_i^ATT   = T_i + (1 - T_i) * e(X_i) / (1 - e(X_i))    (target: ATT)
    w_i^ATE   = T_i / e(X_i) + (1 - T_i) / (1 - e(X_i))    (target: ATE)

    e(X_i)   = Pr(T_i = 1 | X_i)                            propensity
    stabilized: multiply weights by Pr(T = 1) or Pr(T = 0) marginal

ATE estimators
    IPTW (Horvitz-Thompson):  mean(w_i T_i Y_i - w_i (1 - T_i) Y_i)
    Hajek (self-normalized):  sum(w_i T_i Y_i) / sum(w_i T_i)  -  sum(w_i (1-T_i) Y_i) / sum(w_i (1-T_i))
    Hajek is more stable when weights are noisy.

AIPW / Doubly-Robust estimator (Robins-Rotnitzky-Zhao 1994)
    Combines outcome regression mu_0, mu_1 with IPW:
        ATE_DR = mean(mu_1(X) - mu_0(X) + T (Y - mu_1(X)) / e(X) - (1-T) (Y - mu_0(X)) / (1 - e(X)))
    Consistent if EITHER the outcome model OR the propensity model is
    correctly specified (double robustness).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _fit_propensity(X, T):
    def neg_ll(beta):
        z = X @ beta
        return -np.sum(T * z - np.logaddexp(0, z))
    res = minimize(neg_ll, np.zeros(X.shape[1]), method="BFGS")
    return 1 / (1 + np.exp(-(X @ res.x)))


def iptw(X, T, Y, target: str = "ATE", stabilize: bool = True) -> dict:
    """IPTW ATE or ATT with optional stabilized weights + Hajek estimator."""
    X = np.asarray(X, dtype=float); T = np.asarray(T, dtype=int); Y = np.asarray(Y, dtype=float)
    e = _fit_propensity(X, T)
    e = np.clip(e, 1e-3, 1 - 1e-3)
    if target == "ATE":
        w = T / e + (1 - T) / (1 - e)
    elif target == "ATT":
        w = T + (1 - T) * e / (1 - e)
    else:
        raise ValueError("target must be 'ATE' or 'ATT'")
    if stabilize:
        p_T = T.mean()
        w = w * np.where(T == 1, p_T, 1 - p_T)
    ht = float(np.mean(w * T * Y - w * (1 - T) * Y))
    # Hajek (self-normalized)
    hajek = float((np.sum(w * T * Y) / np.sum(w * T)) - (np.sum(w * (1 - T) * Y) / np.sum(w * (1 - T))))
    return {"target": target, "IPTW_HorvitzThompson": ht,
            "IPTW_Hajek": hajek,
            "max_weight": float(w.max()), "mean_weight": float(w.mean()),
            "stabilized": bool(stabilize),
            "method": "Inverse-probability-of-treatment weighting"}


def aipw(X, T, Y, target: str = "ATE") -> dict:
    """AIPW / doubly-robust ATE estimator with linear outcome model + logistic PS."""
    X = np.asarray(X, dtype=float); T = np.asarray(T, dtype=int); Y = np.asarray(Y, dtype=float)
    e = _fit_propensity(X, T)
    e = np.clip(e, 1e-3, 1 - 1e-3)
    # Outcome regression per treatment arm (OLS)
    def _fit_arm(mask):
        beta, *_ = np.linalg.lstsq(X[mask], Y[mask], rcond=None)
        return X @ beta
    mu1 = _fit_arm(T == 1); mu0 = _fit_arm(T == 0)
    ate_dr = float(np.mean(mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)))
    return {"ATE_DR": ate_dr,
            "mu1_bar": float(mu1.mean()), "mu0_bar": float(mu0.mean()),
            "method": "AIPW / doubly-robust ATE"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800
    x1 = rng.normal(size=n); x2 = rng.normal(size=n)
    logit_e = -0.5 + 1 * x1 - 0.5 * x2
    T = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit_e))).astype(int)
    Y = 1 + 2 * T + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])

    print("=== Naive difference in means (biased) ===")
    print(f"  Y_treat - Y_control = {Y[T == 1].mean() - Y[T == 0].mean():.3f}")

    print("\n=== IPTW ATE (stabilized) ===")
    r = iptw(X, T, Y, target="ATE", stabilize=True)
    print(f"  Horvitz-Thompson = {r['IPTW_HorvitzThompson']:.3f}")
    print(f"  Hajek            = {r['IPTW_Hajek']:.3f}   (true 2.0)")
    print(f"  max weight       = {r['max_weight']:.3f}")

    print("\n=== IPTW ATT (unstabilized) ===")
    r = iptw(X, T, Y, target="ATT", stabilize=False)
    print(f"  Hajek ATT = {r['IPTW_Hajek']:.3f}")

    print("\n=== AIPW / Doubly-Robust ATE ===")
    r = aipw(X, T, Y)
    print(f"  ATE_DR = {r['ATE_DR']:.3f}  (true 2.0)")

    print("\n--- library cross-check (R WeightIt, PSweight, DoubleML) ---")

"""Regression Discontinuity Design (Reference §15.9).

Treatment assigned by a KNOWN CUTOFF on a continuous 'running variable' r:
    Sharp RDD: T = 1 if r >= c, else 0                    (deterministic)
    Fuzzy RDD: Pr(T = 1) jumps at r = c but not to 1     (probabilistic)

Under continuity of E[Y(0) | R] and E[Y(1) | R] at c, the treatment effect
at the cutoff is identified by the DISCONTINUITY:

    Sharp:  tau_SRD = lim_{r -> c+} E[Y|R=r]  -  lim_{r -> c-} E[Y|R=r]
    Fuzzy:  tau_FRD = numerator / denominator (IV-style rescaling by prob-of-treat jump)

Local-linear estimator (Hahn-Todd-Van der Klaauw 2001)
    Fit two local-linear regressions on either side of the cutoff, using
    a KERNEL-weighted subsample within a bandwidth h.  Report the fitted
    intercepts and their difference.

Bandwidth: Imbens-Kalyanaraman (2012) plug-in or cross-validation.
Use a triangular kernel for optimal MSE.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _triangular_kernel(u):
    return np.maximum(1 - np.abs(u), 0)


def sharp_rdd(r, y, cutoff: float = 0.0, bandwidth: float = None,
              kernel: str = "triangular") -> dict:
    """Local-linear sharp RDD estimator with triangular kernel weights."""
    r = np.asarray(r, dtype=float); y = np.asarray(y, dtype=float)
    if bandwidth is None:
        bandwidth = 1.06 * np.std(r) * len(r) ** (-1 / 5)     # rule-of-thumb (Silverman)
    r_c = r - cutoff
    if kernel == "triangular":
        w = _triangular_kernel(r_c / bandwidth)
    else:
        w = (np.abs(r_c) <= bandwidth).astype(float)
    def _lm(mask):
        X = np.column_stack([np.ones(mask.sum()), r_c[mask]])
        wv = w[mask]
        WX = X * wv[:, None]
        beta = np.linalg.solve(X.T @ WX, X.T @ (wv * y[mask]))
        return beta
    beta_r = _lm((r_c >= 0) & (w > 0))
    beta_l = _lm((r_c < 0) & (w > 0))
    tau = float(beta_r[0] - beta_l[0])
    return {"tau_SRD": tau, "bandwidth": float(bandwidth),
            "intercept_right": float(beta_r[0]),
            "intercept_left": float(beta_l[0]),
            "slope_right": float(beta_r[1]),
            "slope_left": float(beta_l[1]),
            "n_right": int(((r_c >= 0) & (w > 0)).sum()),
            "n_left": int(((r_c < 0) & (w > 0)).sum()),
            "method": "Sharp RDD local-linear (triangular kernel)"}


def fuzzy_rdd(r, y, T, cutoff: float = 0.0, bandwidth: float = None) -> dict:
    """Fuzzy RDD: rescale the outcome jump by the probability-of-treatment jump."""
    r = np.asarray(r, dtype=float); T = np.asarray(T, dtype=float)
    y_res = sharp_rdd(r, y, cutoff, bandwidth)
    t_res = sharp_rdd(r, T, cutoff, bandwidth)
    if abs(t_res["tau_SRD"]) < 1e-6:
        return {"tau_FRD": float("nan"), "note": "no first-stage treatment jump"}
    return {"tau_FRD": float(y_res["tau_SRD"] / t_res["tau_SRD"]),
            "outcome_jump": y_res["tau_SRD"],
            "treatment_jump": t_res["tau_SRD"],
            "bandwidth": y_res["bandwidth"],
            "method": "Fuzzy RDD (Wald / IV formulation)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    r = rng.uniform(-2, 2, n)
    T = (r >= 0).astype(int)
    # y(0) = 1 + 0.5 r,  y(1) = 3 + 0.5 r  -> tau = 2 at cutoff
    Y = 1 + 0.5 * r + 2 * T + rng.normal(0, 0.6, n)

    print("=== Sharp RDD (true tau = 2) ===")
    for h in (0.4, 0.8, 1.5, None):
        rr = sharp_rdd(r, Y, cutoff=0.0, bandwidth=h)
        print(f"  bandwidth = {rr['bandwidth']:.3f}: tau = {rr['tau_SRD']:.3f}   "
              f"n_right = {rr['n_right']}, n_left = {rr['n_left']}")

    print("\n=== Fuzzy RDD ===")
    # simulate fuzzy assignment
    r_f = rng.uniform(-2, 2, n)
    T_f = ((r_f >= 0) & (rng.uniform(size=n) < 0.75)).astype(int)  # 75% comply
    Y_f = 1 + 0.5 * r_f + 2 * T_f + rng.normal(0, 0.6, n)
    rr = fuzzy_rdd(r_f, Y_f, T_f, cutoff=0.0, bandwidth=0.8)
    print(f"  tau_FRD = {rr['tau_FRD']:.3f}   (true = 2.0)")
    print(f"  outcome_jump = {rr['outcome_jump']:.3f}, treatment_jump = {rr['treatment_jump']:.3f}")

    print("\n--- library cross-check (rdrobust / rddensity in R; rdd Python) ---")
    print("  R: rdrobust::rdrobust(y = Y, x = r, c = 0)")

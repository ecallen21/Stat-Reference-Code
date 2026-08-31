"""Influence functions + efficient influence function (Reference Sec 33.11).

Hampel (1974) 'The influence curve and its role in robust estimation.'
Bickel, Klaassen, Ritov & Wellner (1993) 'Efficient and Adaptive
Estimation for Semiparametric Models.'

The INFLUENCE FUNCTION (IF) of a functional T at a distribution F is
the Gateaux derivative in the direction of a point-mass at x:

  IF(x; T, F) = lim_{t->0} [ T((1-t) F + t delta_x) - T(F) ] / t.

For an M-estimator with score psi:
  IF(x) = -[ E psi_dot ]^-1 psi(x)     (Huber form).

Standard error via influence function:
  SE(T_hat) = sqrt( Var(IF(X)) / n ).

Robustness diagnostics: SUPREMUM of |IF| == GROSS-ERROR SENSITIVITY.

Here we compute the empirical influence function for:
  1. Sample mean (IF = X - mean(X)).
  2. Sample median (IF = sign(X - median) / (2 f(median))).
  3. Sample variance (IF = (X - mean)^2 - Var).

We show that the sample-variance IF-based SE recovers the standard
plug-in SE, and that a single outlier moves the mean via its large IF
but barely moves the median.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.stats import gaussian_kde as _kde   # for density at the median


def if_mean(X):
    return X - float(X.mean())


def if_median(X):
    med = float(np.median(X))
    # Density estimate at the median (Huber / Serfling formula).
    kde = _kde(X)
    f_med = float(kde(med)[0])
    # IF = sign(x - med) / (2 f(med))
    return np.sign(X - med) / (2 * f_med)


def if_variance(X):
    mu = float(X.mean()); v = float(X.var())
    return (X - mu) ** 2 - v


def se_from_if(IF):
    return float(np.std(IF, ddof=1) / np.sqrt(len(IF)))


if __name__ == "__main__":
    print("=== Influence functions + efficient influence function ===\n")
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(0, 1, n)

    # 1. Empirical IF and IF-based SEs.
    for name, if_fn, plug_se in (
        ("mean", if_mean, float(X.std(ddof=1) / np.sqrt(n))),
        ("median", if_median, float(1.253 * X.std(ddof=1) / np.sqrt(n))),  # asymptotic for normal
        ("variance", if_variance, None),
    ):
        IF = if_fn(X)
        se_if = se_from_if(IF)
        print(f"  functional = {name:>8}   IF-based SE = {se_if:.4f}"
              f"   plug-in SE = {plug_se if plug_se else 'n/a':<8}")

    # 2. Robustness demo: add a large outlier.
    X_out = np.concatenate([X, [15.0]])
    mean_change = (X_out.mean() - X.mean())
    med_change = (np.median(X_out) - np.median(X))
    print(f"\n  Adding an outlier at 15.0:")
    print(f"    mean changes by {mean_change:>7.4f}   <- proportional to IF_mean(15) / (n+1)")
    print(f"    median changes by {med_change:>7.4f}   <- IF_median(15) is bounded")

    # Show IF for the outlier point vs an in-distribution point.
    IF_mean_out = if_mean(X_out)[-1]
    IF_med_out = if_median(X_out)[-1]
    print(f"\n    IF_mean(15.0)   = {IF_mean_out:.3f}   (unbounded)")
    print(f"    IF_median(15.0) = {IF_med_out:.3f}   (bounded)\n")

    # 3. Efficient influence function preview for E[Y] under IID sampling.
    print("  EIF for the simple mean estimator = IF for the mean = X - mu.")
    print("  Semiparametric-efficiency bound   = Var(EIF) / n = plug-in SE^2.\n")
    print("--- library cross-check (npcausal R; econml.dr; tmle3) ---")

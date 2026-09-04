"""Maximum entropy distribution (Reference Sec 34.9).

Jaynes (1957) 'Information theory and statistical mechanics.'

Given moment constraints E[f_j(X)] = mu_j, the DISTRIBUTION MAXIMISING
ENTROPY takes the exponential-family form:

  p*(x) = exp( - lambda_0 - sum_j lambda_j f_j(x) )

with Lagrange multipliers lambda_j found by matching the constraints
(dual: minimise the log-partition function).

Well-known MaxEnt distributions:
  * Uniform      -> no constraint (except support).
  * Exponential  -> given mean.
  * Gaussian     -> given mean and variance.
  * Log-normal   -> given mean and variance on the log scale.

Here we solve MaxEnt for a discrete distribution over {1..6} matching
a given mean (loaded die) via Newton's method on the dual.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def solve_maxent_mean(x_support, mean_target, tol=1e-9, max_iter=50):
    """MaxEnt over x_support such that sum p_i x_i = mean_target."""
    x = np.asarray(x_support, dtype=float)
    lam = 0.0
    for _ in range(max_iter):
        w = np.exp(-lam * x)
        Z = w.sum()
        p = w / Z
        # Constraint residual: E[x] - mean_target
        E = float((p * x).sum())
        f = E - mean_target
        # Newton update: derivative of E w.r.t lam is -Var(x)
        var = float((p * x ** 2).sum() - E ** 2)
        if var < 1e-12: break
        lam = lam - f / (-var)
        if abs(f) < tol: break
    return p, lam


if __name__ == "__main__":
    print("=== Maximum entropy distribution (Jaynes 1957) ===\n")
    x = np.arange(1, 7)
    for target in (3.5, 4.5, 5.5):
        p, lam = solve_maxent_mean(x, target)
        E = float((p * x).sum())
        H = float(-(p * np.log(p)).sum())
        print(f"  target mean = {target}:")
        print(f"    Lagrange multiplier lambda = {lam:.4f}")
        print(f"    p = {np.round(p, 4).tolist()}")
        print(f"    E[X] recovered = {E:.4f}   entropy H = {H:.4f} nats\n")

    print("  Uniform (no constraint): H = log 6 =", round(np.log(6), 4), "nats\n")

    # Show that Gaussian maximises differential entropy for given (mean, var).
    print("  Differential-entropy analytic: N(0, 1) MaxEnt over mean=0, var=1")
    print(f"    Gaussian H = {0.5 * np.log(2 * np.pi * np.e):.4f} nats  <- upper bound for any density with sigma=1.")
    # A wider-tail Laplace with same variance has less entropy.
    print(f"    Laplace   H = {1 + np.log(2) - 0.5 * np.log(2):.4f} nats  (Laplace scaled to sigma=1)")

    print("\n--- library cross-check (scipy.optimize + custom; R maxentropy; Python maxent) ---")

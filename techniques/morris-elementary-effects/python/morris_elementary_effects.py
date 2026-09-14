"""Morris Elementary Effects (Reference Sec 47.335).

Morris 1991 Technometrics. Cheap screening sensitivity: for
each factor i, compute the FINITE DIFFERENCE (elementary
effect) at random starting points along a grid:

    EE_i(x) = (f(x + delta * e_i) - f(x)) / delta

Aggregate over r trajectories:
    mu_i*  = mean |EE_i|         (importance)
    sigma_i = std EE_i           (interaction / nonlinearity)

Costs r * (d + 1) evaluations, vs O(d * N) for Sobol.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def morris_ee(f, d, r=20, p_levels=6, rng=None):
    """Radial-style Morris elementary effects with p_levels grid."""
    if rng is None: rng = np.random.default_rng(0)
    delta = p_levels / (2 * (p_levels - 1))
    EEs = np.zeros((r, d))
    for k in range(r):
        # Base point on the grid, integer levels
        base_int = rng.integers(0, p_levels // 2, d)
        base = base_int / (p_levels - 1)
        f_base = f(base)
        perm = rng.permutation(d)
        for j in perm:
            shifted = base.copy(); shifted[j] += delta
            f_shift = f(shifted)
            EEs[k, j] = (f_shift - f_base) / delta
            base = shifted; f_base = f_shift
    mu_star = np.mean(np.abs(EEs), axis=0)
    sigma = np.std(EEs, axis=0)
    return mu_star, sigma


if __name__ == "__main__":
    print("=== Morris Elementary Effects (Morris 1991) ===\n")

    # Toy 4-input function: strong dep on X_1, X_2; nonlinear in X_2; nothing in X_4
    def f(x):
        return 3 * x[0] + 5 * x[1] ** 2 + 0.1 * x[2] * x[0] + 0 * x[3]

    mu_star, sigma = morris_ee(f, d=4, r=20, p_levels=6,
                                 rng=np.random.default_rng(0))
    print(f"  Function: 3*x1 + 5*x2^2 + 0.1*x1*x3 + 0*x4\n")
    print(f"  {'input':>5}  {'mu*':>7}  {'sigma':>7}")
    for i, (m, s) in enumerate(zip(mu_star, sigma), 1):
        print(f"  x{i:<4}  {m:>7.3f}  {s:>7.3f}")

    print(f"\n  Interpretation:")
    print(f"    mu*_1 > 0, sigma_1 small     -> linear main effect")
    print(f"    mu*_2 large, sigma_2 large   -> nonlinear main effect")
    print(f"    mu*_3 small, sigma_3 small   -> negligible (interaction only)")
    print(f"    mu*_4 ~ 0                    -> screen out entirely")

    print(f"\n  Morris is O((d+1) r) evaluations; here 4 factors x 20 trajs = "
          f"{4 * 20 + 20} evaluations. Sobol at same accuracy is O(N (d+2)) - orders of magnitude more.")

    print("\n--- library cross-check (SALib.analyze.morris; sensitivity R) ---")

"""Alpha-Spending Function (Reference Sec 47.258).

Lan & DeMets 1983 Biometrika. FLEXIBLE group-sequential
boundaries: pre-specify how much of the total alpha budget is
"spent" by information fraction t = n_k / N:

    Pocock-like:    alpha(t) = alpha * ln(1 + (e-1) t)
    O'BF-like:      alpha(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t)))
    Linear:         alpha(t) = alpha * t
    Power:          alpha(t) = alpha * t^rho

Advantages over classical Pocock/OBF: interim looks do not have
to be evenly spaced, and K can change mid-trial (must respect
information fraction).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def alpha_obf(t, alpha=0.05):
    """O'Brien-Fleming-like alpha spending."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return 2 * (1 - norm.cdf(z / np.sqrt(t)))


def alpha_pocock(t, alpha=0.05):
    """Pocock-like alpha spending."""
    return alpha * np.log(1 + (np.e - 1) * t)


def alpha_power(t, alpha=0.05, rho=1.0):
    """Power-family alpha spending."""
    return alpha * t ** rho


def compute_boundaries(t_looks, spend_fn, alpha=0.05, n_sim=50_000, rng=None):
    """Convert cumulative spending increments into per-look Z-boundaries."""
    if rng is None: rng = np.random.default_rng(0)
    increments = np.diff(np.concatenate([[0], spend_fn(t_looks, alpha)]))
    K = len(t_looks)
    # Simulate correlated Z-stats under H0: increments in info units
    dn = np.diff(np.concatenate([[0], t_looks]))
    dW = rng.standard_normal((n_sim, K)) * np.sqrt(dn)
    Z = dW.cumsum(axis=1) / np.sqrt(t_looks)
    bounds = np.zeros(K); crossed = np.zeros(n_sim, dtype=bool)
    for k in range(K):
        # Find c_k such that P(|Z_k| > c_k AND not crossed before) = increment
        candidates = np.abs(Z[~crossed, k])
        target_p = increments[k] * n_sim / (n_sim - crossed.sum())
        if target_p >= 1: target_p = 0.9999
        bounds[k] = float(np.quantile(candidates, 1 - target_p))
        crossed |= np.abs(Z[:, k]) > bounds[k]
    return bounds


if __name__ == "__main__":
    print("=== Alpha-Spending Function (Lan & DeMets 1983) ===\n")
    rng = np.random.default_rng(0)

    alpha = 0.05
    t_even = np.array([0.25, 0.50, 0.75, 1.00])
    t_uneven = np.array([0.15, 0.40, 0.80, 1.00])

    print(f"  Alpha = {alpha}, K = 4 looks.\n")

    for name, spend_fn in [("O'BF-like  ", alpha_obf),
                            ("Pocock-like", alpha_pocock),
                            ("Linear     ", lambda t, a=alpha: alpha_power(t, a, 1.0))]:
        cumul = spend_fn(t_even, alpha)
        print(f"  {name}  cumulative alpha at t=0.25/0.50/0.75/1.00: "
              f"{'  '.join(f'{v:.4f}' for v in cumul)}")

    print()
    b_obf = compute_boundaries(t_even, alpha_obf, alpha=alpha, n_sim=200_000, rng=rng)
    b_pocock = compute_boundaries(t_even, alpha_pocock, alpha=alpha, n_sim=200_000, rng=rng)
    print(f"  OBF boundaries at even looks:    {b_obf.round(3).tolist()}")
    print(f"  Pocock boundaries at even looks: {b_pocock.round(3).tolist()}")

    # Uneven schedule (rare but possible with Lan-DeMets)
    b_obf_uneven = compute_boundaries(t_uneven, alpha_obf, alpha=alpha, n_sim=200_000, rng=rng)
    print(f"\n  OBF boundaries at uneven looks (t=0.15,0.40,0.80,1.00):")
    print(f"    {b_obf_uneven.round(3).tolist()}")
    print(f"  Lan-DeMets adjusts the boundary to preserve alpha at ANY look schedule.")

    print("\n--- library cross-check (gsDesign R; ldbounds R; ldbounds Python) ---")

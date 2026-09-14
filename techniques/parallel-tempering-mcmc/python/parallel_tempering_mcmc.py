"""Parallel Tempering / Replica Exchange (Geyer 1991; Earl-Deem 2005).

Run K parallel MCMC chains at temperatures 1 = T_1 < T_2 < ...
targeting pi^(1/T_k). Hotter chains explore more freely; cold
chain is the one of interest.

Every few steps propose swapping states of adjacent chains
with M-H probability:

    alpha = min(1, (pi(x_hot) / pi(x_cold)) ** (1/T_cold - 1/T_hot))

Cures mixing on multi-modal targets that trap plain MH.
"""

import numpy as np    # arrays + random


def target_log(x):
    """Bimodal mixture of two Gaussians at -3 and +3."""
    a = -0.5 * (x + 3) ** 2
    b = -0.5 * (x - 3) ** 2
    return np.logaddexp(a, b) - np.log(2)


def pt_step(x, T, sigma, rng):
    y = x + sigma * rng.standard_normal()
    log_alpha = (target_log(y) - target_log(x)) / T
    if np.log(rng.uniform()) < log_alpha:
        return y, 1
    return x, 0


def parallel_tempering(n_iter, temps, sigma, seed=0):
    rng = np.random.default_rng(seed)
    K = len(temps)
    xs = np.zeros(K)
    samples = np.empty((n_iter, K))
    swap_counts = np.zeros(K - 1, dtype=int)
    swap_tries = np.zeros(K - 1, dtype=int)
    for t in range(n_iter):
        for k in range(K):
            xs[k], _ = pt_step(xs[k], temps[k], sigma, rng)
        k = rng.integers(0, K - 1)
        swap_tries[k] = swap_tries[k] + 1
        log_alpha = (1 / temps[k] - 1 / temps[k + 1]) * (target_log(xs[k + 1]) - target_log(xs[k]))
        if np.log(rng.uniform()) < log_alpha:
            xs[k], xs[k + 1] = xs[k + 1], xs[k]
            swap_counts[k] = swap_counts[k] + 1
        samples[t] = xs
    return samples, swap_counts, swap_tries


def demo():
    print("=== Parallel Tempering (Geyer 1991) ===")
    print("Target: bimodal mixture N(-3) + N(+3), width 1")

    print("\n1. Plain single-chain MH at T=1 (traps in one mode):")
    rng = np.random.default_rng(0)
    x = 0.0
    single = []
    for t in range(20000):
        y = x + 0.5 * rng.standard_normal()
        if np.log(rng.uniform()) < (target_log(y) - target_log(x)):
            x = y
        single.append(x)
    single = np.array(single[2000:])
    left = np.mean(single < 0)
    print(f"   Fraction in left mode : {left:.3f}  (target 0.5)")

    print("\n2. Parallel tempering with 5 chains, T in {1, 2, 4, 8, 16}:")
    temps = [1.0, 2.0, 4.0, 8.0, 16.0]
    samples, swaps, tries = parallel_tempering(20000, temps, sigma=0.5, seed=1)
    cold = samples[2000:, 0]
    left_pt = np.mean(cold < 0)
    print(f"   Cold-chain fraction in left mode : {left_pt:.3f}  (target 0.5)")
    print(f"   Empirical cold-chain mean        : {cold.mean():.3f}  (target 0.0)")
    for k in range(len(temps) - 1):
        rate = swaps[k] / max(1, tries[k])
        print(f"   Swap rate  T={temps[k]:>4.1f} <-> T={temps[k+1]:>4.1f}  = {rate:.2f}")


if __name__ == "__main__":
    demo()

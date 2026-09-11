"""Best-of-N Sampling (Reference Sec 47.196).

Cobbe et al 2021 'Training Verifiers to Solve Math Word Problems';
Nakano et al 2021 'WebGPT'. Simplest test-time scaling for LMs:

    1. Draw N i.i.d. samples from p(y | x) with temperature > 0.
    2. Score each with a REWARD MODEL / verifier.
    3. Return the argmax-scored sample.

Provably improves quality: E[max_i R(y_i)] increases in N. Cheap
to implement; complements RLHF (which changes the sampling
distribution) — best-of-N is a wrapper that keeps the base model
frozen.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def bon_sample(sample_fn, reward_fn, N, rng):
    """Draw N samples, return the one with highest reward."""
    samples = [sample_fn(rng) for _ in range(N)]
    rewards = np.array([reward_fn(s) for s in samples])
    top = int(np.argmax(rewards))
    return samples[top], rewards


def expected_best_of_n_normal(N, mu=0.0, sigma=1.0, n_mc=20000, rng=None):
    """Monte-Carlo E[max of N N(mu, sigma^2)]."""
    if rng is None: rng = np.random.default_rng(0)
    xs = rng.normal(mu, sigma, size=(n_mc, N))
    return float(np.mean(xs.max(axis=1)))


if __name__ == "__main__":
    print("=== Best-of-N Sampling (Cobbe 2021; Nakano 2021 WebGPT) ===\n")
    rng = np.random.default_rng(0)

    # Empirical E[max R] for N = 1 to 128 draws from N(0, 1)
    print("  E[max of N i.i.d. N(0, 1)] (theoretical + Monte-Carlo, 20k reps each):")
    print(f"    {'N':>4}   {'MC E[max R]':>12}   {'theoretical ~sqrt(2 ln N)':>25}")
    for N in [1, 2, 4, 8, 16, 32, 64, 128]:
        mc = expected_best_of_n_normal(N, n_mc=20000, rng=rng)
        theory = np.sqrt(2 * np.log(max(N, 2)))
        print(f"    {N:>4}   {mc:>12.4f}   {theory:>25.4f}")

    # A concrete simulation: LM samples 4-digit numbers, reward = -1 if odd else abs deviation from truth
    truth = 4242
    def sample_fn(rng):
        return int(rng.integers(1000, 9999))
    def reward_fn(y):
        return -abs(y - truth) + (-100 if y % 2 == 1 else 0)     # heavily penalise odd

    print(f"\n  Toy task: sample 4-digit number, reward = -|y - {truth}| (with odd-penalty).")
    print(f"  Best-of-N reward vs N (average over 500 trials):")
    for N in [1, 4, 16, 64, 256]:
        rewards = []
        for _ in range(500):
            _, all_r = bon_sample(sample_fn, reward_fn, N, rng)
            rewards.append(all_r.max())
        print(f"    N = {N:>3}   mean best reward = {np.mean(rewards):>8.1f}")

    print("\n  Beyond a certain N, additional samples yield diminishing returns.")
    print("  For LMs, N = 4-16 is typical; larger N shifts to search methods (ToT, MCTS).")

    print("\n--- library cross-check (transformers.generate num_return_sequences + custom RM Python) ---")

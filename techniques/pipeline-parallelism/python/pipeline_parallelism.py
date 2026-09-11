"""Pipeline Parallelism (Reference Sec 47.176).

Huang et al 2019 'GPipe: Efficient Training of Giant Neural
Networks using Pipeline Parallelism', NeurIPS. Splits a deep model
into K SEQUENTIAL STAGES on K devices; splits a mini-batch into M
MICRO-BATCHES so multiple stages compute concurrently:

    F_1(mb) -> F_2(mb) -> ... F_K(mb) -> B_K -> ... -> B_1
    (M micro-batches keep all K devices busy after warm-up).

Bubble ratio ~ (K - 1) / (M + K - 1); larger M -> less bubble.
1F1B (one-forward-one-backward, Narayanan 2019 PipeDream) reduces
peak activation memory vs pure GPipe.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def simulate_gpipe(K, M):
    """Simulate GPipe schedule: returns (compute-cell, timeline) matrix (device x step)."""
    # F_k(m) at time (m + k - 1); B_k(m) at time (M + K - 1) + (M - m - 1) + (K - k)
    total_steps = 2 * (M + K - 1)                              # forward + backward passes
    schedule = np.full((K, total_steps), "  ", dtype=object)
    for m in range(M):
        for k in range(K):
            schedule[k, m + k] = f"F{m}"                        # forward
    # Backward pass starts after all forward pass finishes
    fwd_end = M + K - 1
    for m in range(M - 1, -1, -1):
        for k in range(K - 1, -1, -1):
            step = fwd_end + (M - 1 - m) + (K - 1 - k)
            schedule[k, step] = f"B{m}"
    return schedule


def bubble_ratio(K, M):
    """Fraction of pipeline time devices are idle in GPipe."""
    busy = 2 * M * K                                            # 1 F + 1 B per (m, k)
    total = K * (2 * (M + K - 1))
    return 1 - busy / total


if __name__ == "__main__":
    print("=== Pipeline Parallelism / GPipe (Huang et al 2019) ===\n")

    print("  Pipeline bubble ratio vs micro-batch count (K = 4 stages):")
    for M in [1, 2, 4, 8, 16, 32]:
        print(f"    M = {M:2d} micro-batches   bubble = {bubble_ratio(4, M) * 100:5.1f}%")

    print("\n  Schedule (K = 3 stages, M = 3 micro-batches):")
    s = simulate_gpipe(3, 3)
    for k in range(3):
        print(f"    device {k}: {' '.join(s[k])}")

    print("\n  Compute vs comm: each stage owns 1/K of the parameters; only forward")
    print("  activations at stage boundaries cross the network (small vs full weight-sync).")

    # Time-per-step numeric comparison: DDP vs PP for a deep model
    L, d = 32, 512
    fwd_flops_per_layer = 2 * d * d
    bwd_flops_per_layer = 2 * fwd_flops_per_layer
    total_flops = L * (fwd_flops_per_layer + bwd_flops_per_layer)
    # 4-device PP with M=8 micro-batches: per-device flops ~ total / K
    # bubble adds (K-1)/(M+K-1) idle
    for K in [4, 8]:
        M_best = 4 * K
        eff = 1 - bubble_ratio(K, M_best)
        pp_time = total_flops / (K * eff)
        # DDP: all-reduce weights (K * d^2 * L) per step -> comms dominated for big K
        print(f"\n  K = {K} stages, M = {M_best} micro-batches: pipeline efficiency = {eff * 100:.1f}%")

    print("\n--- library cross-check (deepspeed / megatron-lm / torch.distributed.pipeline Python) ---")

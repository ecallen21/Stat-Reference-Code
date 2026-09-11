"""ZeRO - Zero Redundancy Optimizer / DeepSpeed (Reference Sec 47.178).

Rajbhandari, Rasley, Ruwase & He 2020 'ZeRO: Memory Optimizations
Toward Training Trillion Parameter Models', SC. Three stages of
sharding across N data-parallel ranks:

    Stage 1: shard OPTIMIZER STATES (Adam m, v, FP32 master)
    Stage 2: also shard GRADIENTS
    Stage 3: also shard PARAMETERS (equivalent to FSDP)

Rajbhandari 2021 'ZeRO-Infinity': offload to CPU / NVMe for
trillion-parameter models on modest hardware. Each stage adds
comm cost, so real deployments mix ZeRO with tensor / pipeline
parallelism.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def per_rank_memory(n_params, n_ranks, stage=0, offload_optim=False):
    """Per-rank memory (bytes) for Adam optimiser: FP16 params + grads,
    FP32 master + m + v.
    """
    fp16 = 2; fp32 = 4
    if stage == 0:                                              # plain DDP
        params = n_params * fp16
        grads = n_params * fp16
        optim = n_params * fp32 * 3
    elif stage == 1:
        params = n_params * fp16
        grads = n_params * fp16
        optim = (n_params / n_ranks) * fp32 * 3
    elif stage == 2:
        params = n_params * fp16
        grads = (n_params / n_ranks) * fp16
        optim = (n_params / n_ranks) * fp32 * 3
    elif stage == 3:
        params = (n_params / n_ranks) * fp16
        grads = (n_params / n_ranks) * fp16
        optim = (n_params / n_ranks) * fp32 * 3
    if offload_optim:
        optim = 0                                                # offloaded to CPU / NVMe
    return {"params": params, "grads": grads, "optim": optim,
            "total": params + grads + optim}


if __name__ == "__main__":
    print("=== ZeRO Redundancy Optimizer / DeepSpeed (Rajbhandari et al 2020) ===\n")

    n_params = 30_000_000_000
    n_ranks = 16
    print(f"  Model: {n_params / 1e9:.1f}B params, N = {n_ranks} data-parallel ranks (Adam optim)\n")

    ddp_gb = per_rank_memory(n_params, n_ranks, stage=0)["total"] / 1e9
    print(f"  {'config':<25}  {'per-rank total':>16}  {'vs DDP':>10}")
    for stage in [0, 1, 2, 3]:
        m = per_rank_memory(n_params, n_ranks, stage=stage)["total"] / 1e9
        print(f"  ZeRO stage {stage}{'':<15}  {m:>13.1f} GB  {100 * (1 - m / ddp_gb):>8.1f}%")
    # + ZeRO-Offload
    m3o = per_rank_memory(n_params, n_ranks, stage=3, offload_optim=True)["total"] / 1e9
    print(f"  ZeRO stage 3 + offload{'':<3}  {m3o:>13.1f} GB  {100 * (1 - m3o / ddp_gb):>8.1f}%")

    # Sweep n_ranks for ZeRO 3
    print(f"\n  Per-rank total for {n_params / 1e9:.1f}B params under ZeRO stage 3:")
    for r in [1, 4, 8, 16, 32, 64]:
        m = per_rank_memory(n_params, r, stage=3)["total"] / 1e9
        print(f"    N = {r:3d} ranks:  {m:6.1f} GB per rank")

    print("\n  ZeRO-3 = FSDP. Larger N pushes per-rank memory down linearly, at")
    print("  the cost of extra all-gather / reduce-scatter traffic per step.")

    print("\n--- library cross-check (deepspeed / torch.distributed.fsdp / accelerate) ---")

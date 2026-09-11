"""FSDP - Fully Sharded Data Parallel (Reference Sec 47.177).

Zhao et al 2023 'PyTorch FSDP: Experiences on Scaling Fully Sharded
Data Parallel', VLDB. Shards parameters, gradients, AND optimiser
states across N data-parallel ranks. Each rank holds 1/N of each
tensor at rest; all-gathers just-in-time for compute:

    forward layer L:  all-gather params -> compute -> discard.
    backward layer L: all-gather params -> compute grad ->
                          reduce-scatter grad -> discard.

Compared to DDP (each rank keeps a full weight copy): FSDP cuts
memory by N and matches DDP in throughput on modern hardware.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def memory_footprint(n_params, n_ranks, mode="ddp"):
    """Return per-rank (params, grads, optim_states) in bytes, assuming Adam."""
    fp16 = 2; fp32 = 4
    if mode == "ddp":
        params = n_params * fp16
        grads = n_params * fp16
        # Adam optimiser: fp32 master weights + m + v
        optim = n_params * fp32 * 3
    elif mode == "zero_stage_1":                                # shard optimiser only
        params = n_params * fp16
        grads = n_params * fp16
        optim = (n_params / n_ranks) * fp32 * 3
    elif mode == "zero_stage_2":                                # shard optim + grads
        params = n_params * fp16
        grads = (n_params / n_ranks) * fp16
        optim = (n_params / n_ranks) * fp32 * 3
    elif mode == "fsdp":                                        # shard params + grads + optim
        params = (n_params / n_ranks) * fp16
        grads = (n_params / n_ranks) * fp16
        optim = (n_params / n_ranks) * fp32 * 3
    else:
        raise ValueError(mode)
    return {"params": params, "grads": grads, "optim": optim,
            "total": params + grads + optim}


if __name__ == "__main__":
    print("=== FSDP - Fully Sharded Data Parallel (Zhao et al 2023) ===\n")

    # 7B-param LLM (LLaMA-7B-ish); Adam optimiser
    n_params = 7_000_000_000
    print(f"  Model: {n_params / 1e9:.1f}B parameters (FP16 weights + Adam optim states)\n")
    print(f"  {'mode':<15}  {'params':>10}  {'grads':>10}  {'optim':>10}  {'total':>10}  {'reduction':>10}")
    baseline = memory_footprint(n_params, 8, mode="ddp")["total"]
    for mode in ["ddp", "zero_stage_1", "zero_stage_2", "fsdp"]:
        m = memory_footprint(n_params, 8, mode=mode)
        gb = {k: v / 1e9 for k, v in m.items()}
        print(f"  {mode:<15}  {gb['params']:>8.1f}GB  {gb['grads']:>8.1f}GB  "
              f"{gb['optim']:>8.1f}GB  {gb['total']:>8.1f}GB  "
              f"{100 * (1 - m['total'] / baseline):>8.1f}%")

    print("\n  N = 8 ranks; FSDP reduces per-rank memory by ~87.5% (7/8 of DDP shed).")
    print("  Trade: 1.5x - 2x network traffic per step for all-gather / reduce-scatter.")

    # How much can each mode fit on a 40 GB A100?
    print("\n  Max n_params fittable in a 40 GB A100 (single rank of an 8-rank job):")
    for mode in ["ddp", "zero_stage_1", "zero_stage_2", "fsdp"]:
        target_bytes = 40 * 1e9
        # Solve for n_params via binary search
        lo, hi = 1e6, 1e12
        for _ in range(60):
            mid = (lo + hi) / 2
            if memory_footprint(mid, 8, mode=mode)["total"] <= target_bytes:
                lo = mid
            else:
                hi = mid
        print(f"    {mode:<15}: {lo / 1e9:.1f} B params")

    print("\n--- library cross-check (torch.distributed.fsdp / deepspeed ZeRO / accelerate) ---")

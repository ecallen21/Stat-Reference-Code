"""Tensor Parallelism / Megatron-LM (Reference Sec 47.175).

Shoeybi et al 2020 'Megatron-LM: Training Multi-Billion Parameter
Language Models Using Model Parallelism', arXiv. Shards individual
weight matrices ACROSS devices so a single forward / backward
requires just two all-reduce collectives per block:

    Column-parallel linear: W = [W1, W2, ...] split by columns
        Y_i = X @ W_i   (each device);  no all-reduce until later.
    Row-parallel linear: W = [W1; W2; ...] split by rows
        Y = sum_i (X_i @ W_i)   (all-reduce).

For a transformer block:  X -> QKV (col-parallel) -> attention (per-head local) ->
    output-proj (row-parallel, needs all-reduce) -> MLP (col + row, one all-reduce).
Two all-reduces per transformer layer (forward + two per backward).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def column_parallel_linear(X, W_shards):
    """W is [d_in, d_out] split by columns across N shards.

    Each shard i computes Y_i = X @ W_i independently. Concat -> full Y.
    Real implementation runs each on its own device; here we simulate.
    """
    Y_shards = [X @ W_i for W_i in W_shards]
    return np.concatenate(Y_shards, axis=-1)                    # concat over d_out


def row_parallel_linear(X_shards, W_shards):
    """W is [d_in, d_out] split by rows across N shards; X is likewise sharded.

    Each device computes X_i @ W_i, then all-reduce SUMs them -> Y.
    """
    partials = [X_i @ W_i for X_i, W_i in zip(X_shards, W_shards)]
    return sum(partials)                                        # simulated all-reduce sum


def transformer_block_tp(X, WQKV_shards, W_out_shards, W_mlp1_shards, W_mlp2_shards,
                            n_heads_per_shard):
    """One transformer block, tensor-parallel across shards."""
    # 1. QKV projection - column-parallel (each shard owns its heads)
    QKV_shards = column_parallel_linear(X, WQKV_shards)         # shape (T, d_qkv_full)
    # For simplicity we skip actual attention and pass through
    # 2. Output projection - row-parallel (needs all-reduce)
    T, d_total = QKV_shards.shape
    d_per_shard = d_total // len(WQKV_shards)
    X_shards = [QKV_shards[:, i * d_per_shard:(i + 1) * d_per_shard] for i in range(len(W_out_shards))]
    attn_out = row_parallel_linear(X_shards, W_out_shards)      # all-reduce here
    # 3. MLP block: col + row = one all-reduce
    h_shards = column_parallel_linear(attn_out, W_mlp1_shards)
    d_h = h_shards.shape[-1] // len(W_mlp2_shards)
    h_sharded = [h_shards[:, i * d_h:(i + 1) * d_h] for i in range(len(W_mlp2_shards))]
    mlp_out = row_parallel_linear([np.maximum(h, 0) for h in h_sharded], W_mlp2_shards)
    return mlp_out


if __name__ == "__main__":
    print("=== Tensor Parallelism / Megatron-LM (Shoeybi et al 2020) ===\n")
    rng = np.random.default_rng(0)

    d_model, d_qkv, d_ff, T = 32, 32, 128, 16
    X = rng.normal(size=(T, d_model))

    for n_shards in [1, 2, 4]:
        # Split weight matrices across shards
        WQKV_full = rng.normal(scale=0.1, size=(d_model, d_qkv))
        W_out_full = rng.normal(scale=0.1, size=(d_qkv, d_model))
        W_mlp1_full = rng.normal(scale=0.1, size=(d_model, d_ff))
        W_mlp2_full = rng.normal(scale=0.1, size=(d_ff, d_model))
        # Column-parallel: split by output dim
        WQKV_shards = np.split(WQKV_full, n_shards, axis=1)
        W_mlp1_shards = np.split(W_mlp1_full, n_shards, axis=1)
        # Row-parallel: split by input dim
        W_out_shards = np.split(W_out_full, n_shards, axis=0)
        W_mlp2_shards = np.split(W_mlp2_full, n_shards, axis=0)

        Y_tp = transformer_block_tp(X, WQKV_shards, W_out_shards, W_mlp1_shards,
                                        W_mlp2_shards, n_heads_per_shard=None)

        # Reference: single-device compute
        Y_ref = np.maximum(X @ WQKV_full @ W_out_full @ W_mlp1_full, 0) @ W_mlp2_full
        rel_err = float(np.linalg.norm(Y_tp - Y_ref) / np.linalg.norm(Y_ref))

        # Memory per shard
        weight_per_shard_KB = ((d_model * d_qkv + d_qkv * d_model + d_model * d_ff + d_ff * d_model)
                                / n_shards) * 8 / 1024
        print(f"  n_shards = {n_shards:2d}   TP output rel-err vs single-device: {rel_err:.2e}   "
              f"weight per shard = {weight_per_shard_KB:.1f} KB")

    print("\n  Bit-exact equivalence to single-device (up to FP rounding).")
    print("  Memory scales as 1 / n_shards; two all-reduce collectives per layer.")

    print("\n--- library cross-check (megatron-lm / colossalai / accelerate Python) ---")

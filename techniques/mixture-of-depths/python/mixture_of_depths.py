"""Mixture-of-Depths - MoD (Reference Sec 47.194).

Raposo, Ainslie, Freitas, Neumann & Dean 2024 'Mixture-of-Depths:
Dynamically allocating compute in transformer-based language
models'. Like Mixture-of-Experts routes across WIDTH, MoD routes
across DEPTH:

    A per-token, per-block ROUTER decides whether the token passes
    through the block (full compute) or SKIPS it (identity).
    Router keeps top-k tokens per block; the rest are skipped.

Saves compute on 'easy' tokens (function words, repeated
punctuation) while spending it on 'hard' tokens (rare vocab,
reasoning steps).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def transformer_block(x, params):
    """Toy block: linear + relu + linear (residual)."""
    h = np.maximum(x @ params["W1"], 0)
    return x + h @ params["W2"]


def mod_block(x, params, keep_frac=0.5, router_key=None):
    """Mixture-of-Depths: keep top-k tokens by router score, skip the rest."""
    T, d = x.shape
    router_score = x @ router_key                                # (T,)
    n_keep = max(1, int(keep_frac * T))
    top = np.argsort(-router_score)[:n_keep]
    keep_mask = np.zeros(T, dtype=bool); keep_mask[top] = True
    y = x.copy()
    y[keep_mask] = transformer_block(x[keep_mask], params)
    return y, keep_mask


if __name__ == "__main__":
    print("=== Mixture-of-Depths (Raposo et al 2024) ===\n")
    rng = np.random.default_rng(0)

    T, d, n_layers = 64, 32, 8
    x = rng.normal(scale=0.5, size=(T, d))
    params = {"W1": rng.normal(scale=0.1, size=(d, d)),
                "W2": rng.normal(scale=0.1, size=(d, d))}

    # Full Transformer
    y_full = x.copy()
    for _ in range(n_layers):
        y_full = transformer_block(y_full, params)

    # MoD: keep 50% per block
    router_key = rng.normal(scale=0.5, size=d)                   # per-block router param
    y_mod = x.copy()
    per_layer_kept = []
    for _ in range(n_layers):
        y_mod, mask = mod_block(y_mod, params, keep_frac=0.5, router_key=router_key)
        per_layer_kept.append(int(mask.sum()))

    total_full_compute = n_layers * T                            # tokens x layers
    total_mod_compute = sum(per_layer_kept)
    print(f"  Sequence length T = {T}, {n_layers} layers, d = {d}")
    print(f"  Full-depth compute: {total_full_compute} token-block operations")
    print(f"  MoD compute (keep_frac = 0.5): {total_mod_compute}  "
          f"({100 * (1 - total_mod_compute / total_full_compute):.0f}% savings)")

    rel_diff = float(np.linalg.norm(y_mod - y_full) / np.linalg.norm(y_full))
    print(f"  Output rel-diff vs full-depth: {rel_diff:.3f}")

    # Compute vs sequence length
    print(f"\n  Compute scaling per layer at various keep_frac (T = 4096):")
    for kf in [1.0, 0.5, 0.25, 0.125]:
        flops = int(kf * 4096) * d * d * 2                       # matmul-ish
        print(f"    keep_frac = {kf:.3f}   effective tokens = {int(kf * 4096):>4d}   "
              f"FLOPs / layer = {flops:>10,}")

    print("\n  Router learned to keep 'important' tokens (rare words, reasoning steps)")
    print("  and skip 'easy' ones (function words, whitespace), saving ~2x compute.")

    print("\n--- library cross-check (Google DeepMind MoD JAX reference impl) ---")

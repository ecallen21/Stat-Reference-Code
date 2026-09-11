"""PatchTST - Patch Time-Series Transformer (Reference Sec 47.238).

Nie, Nguyen, Sinthong & Kalagnanam 2023 'A Time Series is Worth
64 Words: Long-term Forecasting with Transformers', ICLR. Two
tricks:

    1. PATCHING: split the series into non-overlapping patches
       (like ViT), then apply Transformer to the sequence of
       patch embeddings.
    2. CHANNEL-INDEPENDENT: each variate is forecast by the SAME
       shared Transformer -> parameter-sharing bonus + no
       spurious cross-variate leakage.

Simple, strong long-horizon forecaster; beat Informer /
Autoformer on ETT / Weather / Electricity.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def patchify_series(series, patch_len, stride):
    """Split (T,) into (n_patches, patch_len) via stride."""
    n = (len(series) - patch_len) // stride + 1
    return np.array([series[i * stride:i * stride + patch_len] for i in range(n)])


def linear_head_forecast(patches, W_embed, W_pred, horizon):
    """Toy PatchTST head: linear embed -> mean-pool -> linear predict."""
    embed = patches @ W_embed                                    # (n_patches, d)
    ctx = embed.mean(axis=0)                                     # global pool
    return ctx @ W_pred                                          # (horizon,)


def train_patch_tst(train_series, patch_len, stride, horizon, d=32,
                       lr=0.05, n_iter=500, seed=0):
    rng = np.random.default_rng(seed)
    T = len(train_series) - horizon
    # Windows: for each starting point, take past patches -> horizon target
    W_embed = rng.normal(scale=0.3, size=(patch_len, d))
    W_pred = rng.normal(scale=0.3, size=(d, horizon))
    for it in range(n_iter):
        start = rng.integers(0, T - patch_len * 6)
        past = train_series[start:start + patch_len * 6]
        target = train_series[start + patch_len * 6:start + patch_len * 6 + horizon]
        if len(target) < horizon: continue
        patches = patchify_series(past, patch_len, stride)
        pred = linear_head_forecast(patches, W_embed, W_pred, horizon)
        err = pred - target
        # Grad step (crude)
        ctx = (patches @ W_embed).mean(axis=0)
        W_pred -= lr * np.outer(ctx, err) / horizon
    return W_embed, W_pred


if __name__ == "__main__":
    print("=== PatchTST (Nie et al 2023 ICLR) ===\n")
    rng = np.random.default_rng(0)

    T = 400; horizon = 20; patch_len = 8; stride = 8
    t = np.arange(T)
    series = 3 * np.sin(2 * np.pi * t / 24) + 0.02 * t + 0.5 * rng.normal(size=T)

    W_embed, W_pred = train_patch_tst(series[:-horizon], patch_len, stride,
                                             horizon, d=32, n_iter=1500, seed=0)

    # Predict last horizon
    past = series[-horizon - patch_len * 6:-horizon]
    patches = patchify_series(past, patch_len, stride)
    forecast = linear_head_forecast(patches, W_embed, W_pred, horizon)
    actual = series[-horizon:]
    mse = float(np.mean((forecast - actual) ** 2))
    print(f"  T = {T}, patch_len = {patch_len}, stride = {stride}")
    print(f"  {len(patches)} patches from {len(past)}-step past window")
    print(f"  PatchTST forecast MSE:      {mse:.4f}")
    naive = np.repeat(series[-horizon - 1], horizon)
    print(f"  Naive last-value MSE:       {float(np.mean((naive - actual) ** 2)):.4f}")

    # Show the patching pipeline explicitly
    print(f"\n  Patching pipeline:")
    for i, p in enumerate(patches[:3]):
        print(f"    patch {i}: {np.round(p, 2).tolist()}")
    print(f"  Each patch maps to a d={32}-dim token embedding; sequence goes into")
    print(f"  a Transformer encoder (this toy uses mean-pool, hence the weaker result).")

    print("\n--- library cross-check (thuml/PatchTST; neuralforecast.PatchTST; darts) ---")

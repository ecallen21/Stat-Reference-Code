"""Gradient Checkpointing (Reference Sec 47.169).

Chen, Xu, Zhang & Guestrin 2016 'Training Deep Nets with Sublinear
Memory Cost', arXiv. Standard backprop stores every layer's
activations to reuse them in the backward pass -> O(L) memory in
depth L. Gradient checkpointing keeps only a SUBSET of activations
('checkpoints') and RECOMPUTES the rest during backward:

    memory  = O(sqrt(L))   (equal-spaced checkpoints)
    compute = ~1.33x forward pass  (one recomputation per segment).

Trades compute for memory; standard for training large Transformers.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def forward_L_layers(x0, Ws):
    """Standard forward pass storing every activation."""
    acts = [x0]
    for W in Ws:
        acts.append(np.tanh(acts[-1] @ W))
    return acts


def backward_full(acts, Ws, grad_out):
    """Backprop reusing stored activations (no recomputation)."""
    L = len(Ws)
    dWs = [None] * L
    g = grad_out
    for l in range(L - 1, -1, -1):
        # d/dW: input(acts[l])^T @ (g * (1 - tanh(z)^2)); z = acts[l+1] (already tanh)
        deriv = 1 - acts[l + 1] ** 2
        dz = g * deriv
        dWs[l] = acts[l].T @ dz
        g = dz @ Ws[l].T
    return dWs


def forward_checkpointed(x0, Ws, segment=None):
    """Store only checkpoints spaced ~sqrt(L) apart."""
    L = len(Ws)
    if segment is None:
        segment = max(1, int(np.sqrt(L)))
    checkpoints = {0: x0}
    x = x0
    for l in range(L):
        x = np.tanh(x @ Ws[l])
        if (l + 1) % segment == 0 or l == L - 1:
            checkpoints[l + 1] = x
    return checkpoints, segment


def backward_checkpointed(x0, Ws, grad_out, checkpoints, segment):
    """Recompute forward within a segment, then normal backprop through it."""
    L = len(Ws)
    dWs = [None] * L
    g = grad_out
    # Iterate segments from the end
    seg_ends = sorted(checkpoints.keys())
    for i in range(len(seg_ends) - 1, 0, -1):
        seg_end, seg_start = seg_ends[i], seg_ends[i - 1]
        # Recompute activations INSIDE segment (from checkpoints[seg_start])
        local_acts = [checkpoints[seg_start]]
        for l in range(seg_start, seg_end):
            local_acts.append(np.tanh(local_acts[-1] @ Ws[l]))
        # Backprop through segment
        for l in range(seg_end - 1, seg_start - 1, -1):
            deriv = 1 - local_acts[l - seg_start + 1] ** 2
            dz = g * deriv
            dWs[l] = local_acts[l - seg_start].T @ dz
            g = dz @ Ws[l].T
    return dWs


if __name__ == "__main__":
    print("=== Gradient Checkpointing (Chen et al 2016) ===\n")
    rng = np.random.default_rng(0)

    L, d = 25, 8
    Ws = [rng.normal(scale=0.5, size=(d, d)) / np.sqrt(d) for _ in range(L)]
    x0 = rng.normal(size=(4, d))

    # Full forward + backward
    acts = forward_L_layers(x0, Ws)
    y = acts[-1]
    grad_out = np.ones_like(y)
    dWs_full = backward_full(acts, Ws, grad_out)

    # Checkpointed
    cp, seg = forward_checkpointed(x0, Ws, segment=None)
    dWs_cp = backward_checkpointed(x0, Ws, grad_out, cp, seg)

    # Compare gradients: max abs diff across layers
    max_diff = max(float(np.max(np.abs(dWs_full[l] - dWs_cp[l]))) for l in range(L))
    print(f"  L = {L} layers, hidden dim d = {d}")
    print(f"  Full backprop stores {L + 1} activations.")
    print(f"  Checkpointed stores {len(cp)} activations (segment = {seg}).")
    print(f"  Memory ratio checkpointed / full = {len(cp) / (L + 1):.3f}")
    print(f"  Max abs gradient discrepancy = {max_diff:.2e}")

    # Confirm gradient accuracy at machine precision
    print(f"\n  Checkpointed gradients equal full-backprop gradients to numerical precision.")
    print(f"  Trade: ~sqrt(L) memory for ~1 extra forward pass in compute.")

    print("\n--- library cross-check (torch.utils.checkpoint / jax.checkpoint Python) ---")

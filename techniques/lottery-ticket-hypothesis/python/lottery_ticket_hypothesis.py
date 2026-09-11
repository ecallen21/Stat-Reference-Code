"""Lottery Ticket Hypothesis (Reference Sec 47.161).

Frankle & Carbin 2019 'The Lottery Ticket Hypothesis: Finding
Sparse, Trainable Neural Networks', ICLR. Iterative magnitude
pruning finds a sparse SUBNETWORK ('winning ticket') that, when
reset to its ORIGINAL random init, trains to the dense-net
accuracy in equal (or fewer) iterations:

    1. Init theta_0; train dense network to theta_T.
    2. Prune p% of smallest-magnitude weights -> mask m.
    3. Reset unpruned weights to their theta_0 values.
    4. Retrain masked network from theta_0.
    5. Repeat 2-4 for target sparsity.

Winning tickets exist only at their ORIGINAL init (random reinit
does not work), suggesting the init encodes structural inductive
bias for the task.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def train_masked_linear(X, y, mask, W_init, lr=0.05, n_iter=800):
    """Train a linear regression with fixed sparsity mask, starting from W_init."""
    W = W_init.copy() * mask
    for it in range(n_iter):
        pred = X @ W
        grad = X.T @ (pred - y) / len(X) * mask
        W = W - lr * grad
        W = W * mask                                            # enforce mask
    return W


if __name__ == "__main__":
    print("=== Lottery Ticket Hypothesis (Frankle & Carbin 2019) ===\n")
    rng = np.random.default_rng(0)

    # Sparse regression: only 5 of 100 features truly matter
    n, d = 400, 100
    true_idx = rng.choice(d, size=5, replace=False)
    W_true = np.zeros(d); W_true[true_idx] = rng.normal(scale=2.0, size=5)
    X = rng.normal(size=(n, d))
    y = X @ W_true + rng.normal(scale=0.3, size=n)

    W_init = rng.normal(scale=0.3, size=d)                     # theta_0
    mask_full = np.ones(d)

    # Dense training
    W_dense = train_masked_linear(X, y, mask_full, W_init, lr=0.02, n_iter=1500)
    mse_dense = float(np.mean((X @ W_dense - y) ** 2))
    print(f"  Dense-net MSE (all {d} weights):    {mse_dense:.4f}")

    # Iterative magnitude pruning
    mask = mask_full.copy()
    W_last = W_dense
    for prune_round in [1, 2, 3, 4]:
        # Prune p% of smallest remaining |W|
        active = np.where(mask > 0)[0]
        n_keep = max(int(0.5 * len(active)), 5)
        keep_idx = active[np.argsort(-np.abs(W_last[active]))[:n_keep]]
        new_mask = np.zeros(d); new_mask[keep_idx] = 1
        # Reset unpruned to theta_0, retrain
        W_ticket = train_masked_linear(X, y, new_mask, W_init, lr=0.02, n_iter=1500)
        mse_ticket = float(np.mean((X @ W_ticket - y) ** 2))
        # Also train with RANDOM reinit as control (Frankle-Carbin's negative check)
        W_rand_init = rng.normal(scale=0.3, size=d)
        W_random = train_masked_linear(X, y, new_mask, W_rand_init, lr=0.02, n_iter=1500)
        mse_random = float(np.mean((X @ W_random - y) ** 2))
        sparsity = 1 - new_mask.sum() / d
        recovered = float(new_mask[true_idx].sum())
        print(f"  Round {prune_round}  sparsity = {sparsity * 100:.1f}%  "
              f"kept {int(new_mask.sum())}/{d}  true-features kept = {int(recovered)}/5")
        print(f"    Winning ticket (reset to theta_0):    MSE = {mse_ticket:.4f}")
        print(f"    Random reinit control (same mask):    MSE = {mse_random:.4f}")
        mask, W_last = new_mask, W_ticket

    print("\n  Note: linear regression is convex so winning-ticket-init and random-reinit")
    print("  converge to the same point given the same mask. The paper's key finding")
    print("  (winning tickets beat random reinit) surfaces on non-convex NNs, where")
    print("  the mask + init together encode task-specific structure.")

    print("\n--- library cross-check (torch.nn.utils.prune Python; sparse R via 'sparsevar') ---")

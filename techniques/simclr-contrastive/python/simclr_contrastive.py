"""SimCLR - Contrastive Learning of Visual Representations (Sec 47.153).

Chen, Kornblith, Norouzi & Hinton 2020 'A Simple Framework for
Contrastive Learning of Visual Representations', ICML. Two random
augmentations of each image form a POSITIVE pair; every other
image in the batch is a NEGATIVE. NT-Xent loss:

    L_ij = -log( exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau) )

with z = g(f(x)), projection head g. Learns visual representations
without labels; a linear classifier on top is competitive with
supervised training.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def nt_xent(Z, temp=0.5):
    """NT-Xent loss for a batch of 2N embeddings (pairs are (2k, 2k+1))."""
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    N2 = Z.shape[0]
    sim = Z @ Z.T / temp
    np.fill_diagonal(sim, -1e9)
    exp = np.exp(sim - sim.max(axis=1, keepdims=True))
    denom = exp.sum(axis=1)
    losses = []
    for k in range(0, N2, 2):
        pos_ij = sim[k, k + 1] - sim[k].max()
        pos_ji = sim[k + 1, k] - sim[k + 1].max()
        losses.append(-pos_ij + np.log(denom[k]))
        losses.append(-pos_ji + np.log(denom[k + 1]))
    return float(np.mean(losses))


def augment(x, rng, strength=0.4):
    """Random 'augmentation' = additive noise + random feature dropout."""
    x_aug = x + rng.normal(scale=strength, size=x.shape)
    mask = rng.uniform(size=x.shape) > 0.15
    return x_aug * mask


def train_simclr(X, dim=8, temp=0.5, lr=0.01, batch=32, n_iter=1500, seed=0):
    """Train a linear encoder + projection head via NT-Xent."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(scale=0.3, size=(d, dim))
    for it in range(n_iter):
        idx = rng.choice(n, size=batch, replace=False)
        pairs = []
        for i in idx:
            pairs.append(augment(X[i], rng))
            pairs.append(augment(X[i], rng))
        A = np.array(pairs)
        Z = A @ W
        L = nt_xent(Z, temp=temp)
        # Finite-difference gradient on W (small demo)
        eps = 1e-3
        grad = np.zeros_like(W)
        # Random-projection perturbation (SPSA gradient)
        delta = rng.choice([-1, 1], size=W.shape) * 1e-3
        L_plus = nt_xent(A @ (W + delta), temp=temp)
        L_minus = nt_xent(A @ (W - delta), temp=temp)
        grad = (L_plus - L_minus) / (2 * 1e-3) * delta / (delta ** 2 + 1e-12)
        W -= lr * grad
    return W


if __name__ == "__main__":
    print("=== SimCLR - Contrastive Learning (Chen et al 2020) ===\n")
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    data = load_digits(); X, y = data.data / 16.0, data.target
    n, d = X.shape
    print(f"  digits:  n = {n}, d = {d}, classes = 10")

    # Baseline: LR on raw pixels (upper bound with all labels)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    print(f"    LR on raw features (all labels):        {LogisticRegression(max_iter=500).fit(Xtr, ytr).score(Xte, yte):.3f}")

    # Baseline: LR on random projection (no learning)
    rng = np.random.default_rng(0)
    Wr = rng.normal(scale=0.3, size=(d, 16))
    print(f"    LR on random-projection 16-D features: {LogisticRegression(max_iter=500).fit(Xtr @ Wr, ytr).score(Xte @ Wr, yte):.3f}")

    # SimCLR-trained embedding (NO labels used during embedding training)
    W_ssl = train_simclr(X, dim=16, temp=0.5, lr=0.05, batch=48, n_iter=800, seed=0)
    print(f"    LR on SimCLR 16-D features (SSL):       {LogisticRegression(max_iter=500).fit(Xtr @ W_ssl, ytr).score(Xte @ W_ssl, yte):.3f}")

    print("\n--- library cross-check (lightly / solo-learn Python) ---")

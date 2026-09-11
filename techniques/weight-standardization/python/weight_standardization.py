"""Weight Standardization (Reference Sec 47.172).

Qiao et al 2019 'Weight Standardization', arXiv. Normalises the
WEIGHTS of each layer instead of (or in addition to) the
activations:

    W_hat_{i,j} = (W_{i,j} - mu_i) / sigma_i    (per output channel)

Combined with Group Normalization (Wu-He 2018), matches Batch-
Norm's accuracy at micro-batch sizes (1-4). Standard for object
detection / segmentation where large batches are infeasible.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def weight_standardize(W, eps=1e-5):
    """Center + scale each row (output channel) of W."""
    mu = W.mean(axis=1, keepdims=True)
    sigma = W.std(axis=1, keepdims=True)
    return (W - mu) / (sigma + eps)


def batch_norm(x, gamma, beta, eps=1e-5):
    """Standard BatchNorm: normalise across the batch axis."""
    mu = x.mean(axis=0)
    var = x.var(axis=0)
    return (x - mu) / np.sqrt(var + eps) * gamma + beta


def group_norm(x, num_groups, gamma, beta, eps=1e-5):
    """Group Normalization: normalise within groups of features (batch-independent)."""
    N, C = x.shape
    G = num_groups
    x_g = x.reshape(N, G, C // G)
    mu = x_g.mean(axis=2, keepdims=True)
    var = x_g.var(axis=2, keepdims=True)
    x_g = (x_g - mu) / np.sqrt(var + eps)
    return x_g.reshape(N, C) * gamma + beta


def train_linear_head(X_features, y, lr=0.05, n_iter=300, seed=0):
    rng = np.random.default_rng(seed)
    d = X_features.shape[1]
    W = rng.normal(scale=0.1, size=d)
    for it in range(n_iter):
        p = 1 / (1 + np.exp(-X_features @ W))
        grad = X_features.T @ (p - y) / len(X_features)
        W -= lr * grad
    return W


if __name__ == "__main__":
    print("=== Weight Standardization (Qiao et al 2019) ===\n")
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=800, n_features=32, n_informative=15,
                                  random_state=0)
    y = y.astype(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    # Random linear feature extractor with / without weight standardisation
    W_hidden = rng.normal(scale=1.0, size=(32, 32))
    W_hidden_ws = weight_standardize(W_hidden)
    gamma = np.ones(32); beta = np.zeros(32)

    # Compute per-channel statistics of features (post-hidden, pre-nonlinearity)
    Ftr_plain = Xtr @ W_hidden.T
    Ftr_ws = Xtr @ W_hidden_ws.T

    print(f"  Per-channel feature std (train, plain W):     "
          f"mean = {Ftr_plain.std(axis=0).mean():.3f}, range = "
          f"[{Ftr_plain.std(axis=0).min():.3f}, {Ftr_plain.std(axis=0).max():.3f}]")
    print(f"  Per-channel feature std (train, standardised): "
          f"mean = {Ftr_ws.std(axis=0).mean():.3f}, range = "
          f"[{Ftr_ws.std(axis=0).min():.3f}, {Ftr_ws.std(axis=0).max():.3f}]")

    # With BN
    Ftr_bn = batch_norm(Ftr_plain, gamma, beta)
    Ftr_ws_bn = batch_norm(Ftr_ws, gamma, beta)
    # With GN (small batch equivalent - useful when BN is unstable)
    Ftr_gn = group_norm(Ftr_plain, num_groups=8, gamma=gamma, beta=beta)
    Ftr_ws_gn = group_norm(Ftr_ws, num_groups=8, gamma=gamma, beta=beta)

    def score(F_tr, F_te):
        W = train_linear_head(F_tr, ytr)
        return float(np.mean((F_te @ W > 0) == (yte > 0.5)))

    Fte_plain = Xte @ W_hidden.T
    Fte_ws = Xte @ W_hidden_ws.T
    Fte_bn = batch_norm(Fte_plain, gamma, beta)
    Fte_ws_bn = batch_norm(Fte_ws, gamma, beta)
    Fte_gn = group_norm(Fte_plain, 8, gamma, beta)
    Fte_ws_gn = group_norm(Fte_ws, 8, gamma, beta)

    print(f"\n  Test-accuracy after LR head on frozen 32-D features:")
    print(f"    plain features:                  {score(Ftr_plain, Fte_plain):.3f}")
    print(f"    + BatchNorm:                    {score(Ftr_bn, Fte_bn):.3f}")
    print(f"    + GroupNorm (small-batch fit):  {score(Ftr_gn, Fte_gn):.3f}")
    print(f"    + Weight Standardisation:       {score(Ftr_ws, Fte_ws):.3f}")
    print(f"    + WS + BN:                       {score(Ftr_ws_bn, Fte_ws_bn):.3f}")
    print(f"    + WS + GN:                       {score(Ftr_ws_gn, Fte_ws_gn):.3f}   (recommended micro-batch combo)")

    print("\n--- library cross-check (kornia / timm weight_std variants Python) ---")

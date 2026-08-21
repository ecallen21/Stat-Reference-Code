"""Convolutional neural network from scratch (Reference §27.2).

2-D convolution layer:
    for each output channel c, spatial position (i, j):
        Y[c, i, j] = sum_{c', p, q} X[c', i+p, j+q] * K[c, c', p, q] + b[c]

Followed by ReLU + max-pooling + flatten + linear.

We implement a small "one convolutional layer + linear classifier" as a
pedagogical demo on 8x8 binary shapes (vertical vs horizontal bars).
Backprop for convolution is another convolution (with the kernel flipped);
we use scipy signal for the convolution forward+backward for compactness.
Serious CNNs use im2col + BLAS or CUDA; from-scratch, this is O(n_out * K^2).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _conv2d_forward(X, K, b):
    """X: (N, C_in, H, W)  K: (C_out, C_in, kH, kW)  b: (C_out,)
    Returns Y: (N, C_out, H - kH + 1, W - kW + 1)."""
    N, C_in, H, W = X.shape; C_out, _, kH, kW = K.shape
    oH = H - kH + 1; oW = W - kW + 1
    Y = np.zeros((N, C_out, oH, oW))
    for i in range(oH):
        for j in range(oW):
            patch = X[:, :, i: i + kH, j: j + kW]           # (N, C_in, kH, kW)
            Y[:, :, i, j] = np.tensordot(patch, K, axes=([1, 2, 3], [1, 2, 3])) + b
    return Y


def _conv2d_backward(dY, X, K):
    """Given dY: (N, C_out, oH, oW), compute (dX, dK, db) using explicit sums."""
    N, C_in, H, W = X.shape; C_out, _, kH, kW = K.shape
    _, _, oH, oW = dY.shape
    dX = np.zeros_like(X); dK = np.zeros_like(K); db = dY.sum(axis=(0, 2, 3))
    for i in range(oH):
        for j in range(oW):
            patch = X[:, :, i: i + kH, j: j + kW]           # (N, C_in, kH, kW)
            # gradients wrt K: sum over N and spatial output positions
            dK += np.einsum("no,ncij->ocij", dY[:, :, i, j], patch)
            # gradients wrt X: accumulate outer product of dY and K
            dX[:, :, i: i + kH, j: j + kW] += np.einsum("no,ocij->ncij", dY[:, :, i, j], K)
    return dX, dK, db


def _max_pool_2x2(X):
    N, C, H, W = X.shape
    X_ = X[:, :, :2 * (H // 2), :2 * (W // 2)]
    X_ = X_.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
    return X_


def _relu(z): return np.maximum(z, 0.0)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=1, keepdims=True)


def fit_small_cnn(X, y, epochs: int = 60, lr: float = 0.05, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N, C_in, H, W = X.shape
    C_out = 4; kH = kW = 3
    K = rng.normal(scale=0.5, size=(C_out, C_in, kH, kW))
    b = np.zeros(C_out)
    pooled = _max_pool_2x2(_relu(_conv2d_forward(X[:1], K, b))).shape
    flat_dim = pooled[1] * pooled[2] * pooled[3]
    C_class = int(y.max()) + 1
    W2 = rng.normal(scale=0.1, size=(flat_dim, C_class))
    b2 = np.zeros(C_class)
    Y = np.zeros((N, C_class)); Y[np.arange(N), y] = 1
    for ep in range(epochs):
        conv = _conv2d_forward(X, K, b)
        act = _relu(conv)
        pooled_v = _max_pool_2x2(act)                       # (N, C_out, oH/2, oW/2)
        flat = pooled_v.reshape(N, -1)
        logits = flat @ W2 + b2
        probs = _softmax(logits)
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()

        # gradient wrt logits, W2, b2
        dlogits = (probs - Y) / N
        dW2 = flat.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dflat = dlogits @ W2.T
        dpooled = dflat.reshape(pooled_v.shape)
        # unpool via arg-max (simplified: pass gradient uniformly through the 2x2)
        # for stability of the demo, replicate to the un-pooled shape divided by 4
        dact = np.repeat(np.repeat(dpooled, 2, axis=2), 2, axis=3) / 4.0
        # crop to conv shape
        dact = dact[:, :, :conv.shape[2], :conv.shape[3]]
        dconv = dact * (conv > 0)
        _, dK, db_c = _conv2d_backward(dconv, X, K)
        W2 -= lr * dW2; b2 -= lr * db2
        K  -= lr * dK; b  -= lr * db_c

    logits = (_max_pool_2x2(_relu(_conv2d_forward(X, K, b))).reshape(N, -1)
              @ W2 + b2)
    acc = float((logits.argmax(axis=1) == y).mean())
    return {"K": K, "b": b, "W2": W2, "b2": b2,
            "train_accuracy": acc, "final_loss": float(loss),
            "method": "1-conv-layer CNN + max-pool + linear + softmax"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 8x8 binary shapes: 0 = vertical bar, 1 = horizontal bar
    def make(kind):
        img = np.zeros((8, 8))
        if kind == 0:                                        # vertical
            col = rng.integers(1, 7); img[:, col: col + 2] = 1
        else:                                                # horizontal
            row = rng.integers(1, 7); img[row: row + 2, :] = 1
        img = img + rng.normal(scale=0.05, size=(8, 8))
        return img
    N = 240
    kinds = rng.integers(0, 2, N)
    X = np.stack([make(int(k)) for k in kinds])[:, None, :, :]     # (N, 1, 8, 8)
    y = kinds.astype(int)                                            # true label = shape kind

    m = fit_small_cnn(X, y, epochs=120, lr=0.05)
    print(f"=== 1-conv-layer CNN on 8x8 bar images (N={N}) ===")
    print(f"  train accuracy = {m['train_accuracy']:.3f}")
    print(f"  final loss     = {m['final_loss']:.4f}")

    print("\n--- library cross-check (pytorch / tensorflow-keras) ---")
    try:
        import torch
        import torch.nn as nn
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        net = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(),
                             nn.MaxPool2d(2), nn.Flatten(),
                             nn.LazyLinear(2))
        opt = torch.optim.Adam(net.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(60):
            opt.zero_grad()
            loss = loss_fn(net(X_t), y_t); loss.backward(); opt.step()
        pred = net(X_t).argmax(dim=1).numpy()
        print(f"  torch CNN train accuracy = {(pred == y).mean():.3f}")
    except ImportError:
        print("  (pytorch not installed)")

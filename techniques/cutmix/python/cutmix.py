"""CutMix (Reference Ch 30 Robustness).

Yun, Han, Chun, Oh, Yoo & Choe (2019) "CutMix: Regularization Strategy to
Train Strong Classifiers with Localizable Features."

Unlike mixup's PIXEL-WISE convex combination, CutMix pastes a RECTANGULAR
patch of image j onto image i and mixes the labels in proportion to the
patch's area:

  lam ~ Beta(alpha, alpha)
  rect area fraction = (1 - lam)
  x_tilde = x_i with a random rect of x_j pasted in
  y_tilde = lam * y_i + (1 - lam) * y_j

Preserves local pixel statistics (unlike mixup's ghosting), while still
regularising the classifier away from over-confidence.

Here we demonstrate the RECTANGULAR MASK MECHANICS on tiny 16x16
'images' and train a linear classifier on flattened pixels for a
synthetic 3-class problem where each class has a distinct location for
its bright region.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def rand_bbox(H, W, lam, rng):
    """CutMix bbox proposal (from Yun 2019)."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio); cut_w = int(W * cut_ratio)
    cy = rng.integers(0, H); cx = rng.integers(0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    return y1, y2, x1, x2


def cutmix_batch(X, Y_one, alpha, rng, H, W):
    n = X.shape[0]
    idx2 = rng.permutation(n)
    lam = rng.beta(alpha, alpha)
    y1, y2, x1, x2 = rand_bbox(H, W, lam, rng)
    # Reshape flat images to (n, H, W); paste patch.
    Xr = X.reshape(n, H, W).copy()
    Xr[:, y1:y2, x1:x2] = X[idx2].reshape(n, H, W)[:, y1:y2, x1:x2]
    # Adjust lam to the true pixel area (may differ due to clipping).
    lam_adj = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)
    Y_mix = lam_adj * Y_one + (1 - lam_adj) * Y_one[idx2]
    return Xr.reshape(n, -1), Y_mix


def train(X, y, K, H, W, use_cutmix=False, alpha=1.0, lr=0.05, epochs=800,
           batch=64, l2=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W_mat = np.zeros((d, K))
    n = X.shape[0]
    Y_one = np.eye(K)[y]
    for _ in range(epochs):
        idx = rng.integers(0, n, batch)
        Xb, Yb = X[idx], Y_one[idx]
        if use_cutmix:
            Xb, Yb = cutmix_batch(Xb, Yb, alpha, rng, H, W)
        p = _softmax(Xb @ W_mat)
        g = Xb.T @ (p - Yb) / batch + l2 * W_mat
        W_mat -= lr * g
    return W_mat


def make_data(rng, n, H, W, K):
    """Each class has a bright 4x4 square at a FIXED corner (0=TL, 1=TR, 2=BL)."""
    y = rng.integers(0, K, n)
    X = np.zeros((n, H, W))
    corners = [(0, 0), (0, W - 4), (H - 4, 0)]
    for i, yi in enumerate(y):
        r, c = corners[yi]
        X[i, r:r + 4, c:c + 4] = 1.0
    X += rng.normal(0, 0.20, X.shape)
    return X.reshape(n, -1), y


if __name__ == "__main__":
    print("=== CutMix (Yun 2019) ===\n")
    rng = np.random.default_rng(0)
    K = 3; H = W = 16
    X_tr, y_tr = make_data(rng, 400, H, W, K)
    X_te, y_te = make_data(rng, 1500, H, W, K)

    # Demo one CutMix operation on the flat batch to show the patch actually swaps.
    idx = rng.permutation(len(X_tr))[:8]
    Y_one = np.eye(K)[y_tr[idx]]
    Xb_mix, Yb_mix = cutmix_batch(X_tr[idx], Y_one, alpha=1.0,
                                    rng=np.random.default_rng(3), H=H, W=W)
    pix_diff = np.mean(np.abs(Xb_mix - X_tr[idx]) > 1e-9)
    print(f"  demo CutMix: fraction of pixels swapped in the batch: {pix_diff:.3f}")
    print(f"  demo CutMix mixed one-hot targets (row 0): {np.round(Yb_mix[0], 3).tolist()}\n")

    W_vanilla = train(X_tr, y_tr, K, H, W, use_cutmix=False)
    W_cutmix = train(X_tr, y_tr, K, H, W, use_cutmix=True, alpha=1.0)

    for name, WM in (("vanilla", W_vanilla), ("cutmix a=1.0", W_cutmix)):
        p = _softmax(X_te @ WM)
        acc = (p.argmax(axis=1) == y_te).mean()
        conf = p.max(axis=1).mean()
        print(f"  {name:15s}  clean_acc={acc:.3f}   mean_conf={conf:.3f}")

    print("\n--- library cross-check (torchvision.transforms.v2.CutMix; timm cutmix helpers) ---")

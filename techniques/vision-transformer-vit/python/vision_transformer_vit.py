"""Vision Transformer - ViT (Reference Sec 47.160).

Dosovitskiy et al 2021 'An Image is Worth 16x16 Words: Transformers
for Image Recognition at Scale', ICLR. Applies a pure Transformer
encoder to sequences of flat image patches:

    1. Split image into P x P patches, linearly project each -> tokens.
    2. Prepend [CLS] token + add learned position embeddings.
    3. L Transformer encoder blocks (multi-head self-attention + MLP).
    4. Linear head on [CLS] token -> class logits.

At sufficient scale (or with strong pre-training) matches / beats
CNNs on ImageNet. The 'inductive bias' of convolutions is traded
for expressive attention over all patches at once.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def patchify(imgs, patch=4):
    """Split (N, H, W) images into (N, num_patches, patch*patch)."""
    N, H, W = imgs.shape
    assert H % patch == 0 and W % patch == 0
    ph, pw = H // patch, W // patch
    x = imgs.reshape(N, ph, patch, pw, patch)
    x = x.transpose(0, 1, 3, 2, 4).reshape(N, ph * pw, patch * patch)
    return x


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, WQ, WK, WV, WO):
    """Single-head self-attention: X (N, T, d) -> (N, T, d)."""
    Q, K, V = X @ WQ, X @ WK, X @ WV
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    A = softmax(scores, axis=-1)
    return (A @ V) @ WO, A


def vit_forward(imgs, params, patch=4, n_classes=10):
    """One-block ViT forward pass with a linear head on the [CLS] token."""
    N, H, W = imgs.shape
    X = patchify(imgs, patch=patch)                             # (N, T, P^2)
    T = X.shape[1]; d = params["W_embed"].shape[1]
    X = X @ params["W_embed"]                                   # linear patch embedding
    cls = np.tile(params["cls"], (N, 1, 1))
    X = np.concatenate([cls, X], axis=1)                        # prepend [CLS]
    X = X + params["pos"][:T + 1][None, :, :]
    attn_out, _ = self_attention(X, params["WQ"], params["WK"], params["WV"], params["WO"])
    X = X + attn_out
    # Simple MLP (2-layer)
    Xm = np.maximum(X @ params["W_mlp1"], 0) @ params["W_mlp2"]
    X = X + Xm
    logits = X[:, 0] @ params["W_head"]                          # [CLS] -> class
    return logits


if __name__ == "__main__":
    print("=== Vision Transformer - ViT (Dosovitskiy et al 2021) ===\n")

    # Toy 8x8 images: 3 classes = horizontal-stripe, vertical-stripe, checker
    rng = np.random.default_rng(0)
    def make_img(cls_):
        img = np.zeros((8, 8))
        if cls_ == 0: img[::2, :] = 1                          # horizontal stripes
        elif cls_ == 1: img[:, ::2] = 1                        # vertical stripes
        else:                                                    # checkerboard
            for i in range(8):
                for j in range(8):
                    if (i + j) % 2 == 0: img[i, j] = 1
        return img + 0.15 * rng.normal(size=(8, 8))

    N_per = 60
    imgs = np.array([make_img(c) for c in [0, 1, 2] for _ in range(N_per)])
    y = np.array([c for c in [0, 1, 2] for _ in range(N_per)])

    # Random-init ViT (frozen) + LR head (representation-learning baseline)
    d = 12; patch = 2; T = (8 // patch) ** 2
    params = {
        "W_embed": rng.normal(scale=0.3, size=(patch * patch, d)),
        "cls": rng.normal(scale=0.3, size=(1, 1, d)),
        "pos": rng.normal(scale=0.3, size=(T + 1, d)),
        "WQ": rng.normal(scale=0.3, size=(d, d)),
        "WK": rng.normal(scale=0.3, size=(d, d)),
        "WV": rng.normal(scale=0.3, size=(d, d)),
        "WO": rng.normal(scale=0.3, size=(d, d)),
        "W_mlp1": rng.normal(scale=0.3, size=(d, d * 2)),
        "W_mlp2": rng.normal(scale=0.3, size=(d * 2, d)),
        "W_head": rng.normal(scale=0.3, size=(d, 3)),
    }

    # Extract representations = [CLS] token embedding after the block
    X = patchify(imgs, patch=patch) @ params["W_embed"]
    cls = np.tile(params["cls"], (len(imgs), 1, 1))
    X = np.concatenate([cls, X], axis=1)
    X = X + params["pos"][None, :, :]
    attn_out, _ = self_attention(X, params["WQ"], params["WK"], params["WV"], params["WO"])
    X = X + attn_out
    reps = np.maximum(X @ params["W_mlp1"], 0) @ params["W_mlp2"]
    features = reps[:, 0]                                        # [CLS] representation

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(features, y, test_size=0.3, random_state=0)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    print(f"  Toy 8x8 shapes, 3 classes, N = {len(imgs)}")
    print(f"  Patch = {patch}x{patch}, T = {T} patches + 1 [CLS], d = {d}")
    print(f"  Random-init ViT [CLS] repr + LR head test acc = {lr.score(Xte, yte):.3f}")

    # Baseline: LR on flattened raw pixels
    Xf = imgs.reshape(len(imgs), -1)
    Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.3, random_state=0)
    lr_flat = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    print(f"  LR on flat 64-dim pixels test acc         = {lr_flat.score(Xte, yte):.3f}")

    print("\n--- library cross-check (timm / transformers.ViTModel Python) ---")

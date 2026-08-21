"""Vision Transformer (Dosovitskiy 2021; Reference §27.x extra).

Turn an image into a sequence of PATCH TOKENS and feed it to a plain
transformer encoder.  ViT pipeline:

  1. Split HxW image into P x P patches -> N = HW/P^2 flat vectors of size P^2 * C.
  2. Linearly project each patch -> patch embedding of size d_model.
  3. Prepend a learnable [CLS] token; add positional embeddings.
  4. Feed the (N + 1) x d_model sequence through a transformer encoder stack.
  5. Linear head on [CLS] output for classification.

This module implements the patch tokeniser + a single transformer encoder
block (reuse the machinery from transformer-encoder) and shows the shapes.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math

import numpy as np    # numerical arrays + linear algebra


def _softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z); return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True); sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def patchify(img, patch: int) -> np.ndarray:
    """img: (H, W, C).  Returns (n_patches, patch*patch*C)."""
    H, W, C = img.shape
    assert H % patch == 0 and W % patch == 0
    n_h = H // patch; n_w = W // patch
    out = img.reshape(n_h, patch, n_w, patch, C).transpose(0, 2, 1, 3, 4)
    return out.reshape(n_h * n_w, patch * patch * C)


def _mha(X, W_q, W_k, W_v, W_o, n_heads):
    n, d_model = X.shape
    d_head = d_model // n_heads
    Q = (X @ W_q).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    K = (X @ W_k).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    V = (X @ W_v).reshape(n, n_heads, d_head).transpose(1, 0, 2)
    outs = []
    for h in range(n_heads):
        s = Q[h] @ K[h].T / math.sqrt(d_head)
        outs.append(_softmax(s, axis=-1) @ V[h])
    return (np.stack(outs).transpose(1, 0, 2).reshape(n, d_model)) @ W_o


def encoder_block(x, params, n_heads):
    xn = layer_norm(x)
    x = x + _mha(xn, params["W_q"], params["W_k"], params["W_v"],
                  params["W_o"], n_heads)
    xn = layer_norm(x)
    x = x + gelu(xn @ params["W_ff1"] + params["b_ff1"]) @ params["W_ff2"] + params["b_ff2"]
    return layer_norm(x)


def vit_forward(img, patch, d_model, n_heads, n_classes, params):
    tokens = patchify(img, patch)                             # (N, patch*patch*C)
    tokens = tokens @ params["W_patch"] + params["b_patch"]   # (N, d_model)
    cls = params["cls"].reshape(1, d_model)
    tokens = np.vstack([cls, tokens])
    tokens = tokens + params["pos"][: len(tokens)]
    for _ in range(params["n_layers"]):
        tokens = encoder_block(tokens, params, n_heads)
    logits = tokens[0] @ params["W_head"] + params["b_head"]
    return logits, tokens


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H = W = 8; C = 3; patch = 4
    n_patches = (H // patch) * (W // patch)                   # 4 patches
    d_model = 32; n_heads = 4; d_ff = 64; n_classes = 3
    n_layers = 2

    d_patch = patch * patch * C
    def _init(shape): return rng.normal(scale=0.1, size=shape)
    params = {
        "W_patch": _init((d_patch, d_model)), "b_patch": np.zeros(d_model),
        "cls": _init(d_model),
        "pos": _init((n_patches + 1, d_model)),
        "W_q": _init((d_model, d_model)), "W_k": _init((d_model, d_model)),
        "W_v": _init((d_model, d_model)), "W_o": _init((d_model, d_model)),
        "W_ff1": _init((d_model, d_ff)), "b_ff1": np.zeros(d_ff),
        "W_ff2": _init((d_ff, d_model)), "b_ff2": np.zeros(d_model),
        "W_head": _init((d_model, n_classes)), "b_head": np.zeros(n_classes),
        "n_layers": n_layers,
    }

    img = rng.normal(size=(H, W, C))
    logits, tokens = vit_forward(img, patch, d_model, n_heads, n_classes, params)
    print(f"=== ViT forward pass (H=W={H}, C={C}, patch={patch}) ===")
    print(f"  # patches (excl. [CLS])       = {n_patches}")
    print(f"  patch input dim               = {d_patch}")
    print(f"  token-sequence shape          = {tokens.shape}   (should be ({n_patches + 1}, {d_model}))")
    print(f"  logits shape                  = {logits.shape}")
    print(f"  logits values                 = {np.round(logits, 3).tolist()}")

    # patch verification: reconstruct one patch from patchify
    p = patchify(img, patch)
    print(f"\n  patchify sanity: img[0:4, 0:4, :].ravel() vs p[0] agree = "
          f"{np.allclose(img[0: patch, 0: patch, :].ravel(), p[0])}")

    print("\n--- library cross-check (torchvision.models.vit_b_16; timm.create_model('vit_base_patch16_224')) ---")

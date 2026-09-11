"""FT-Transformer - Feature-Tokeniser Transformer (Sec 47.242).

Gorishniy, Rubachev, Khrulkov & Babenko 2021 'Revisiting Deep
Learning Models for Tabular Data', NeurIPS. Two ideas:

    1. Feature Tokeniser: each feature (numerical and categorical)
       gets its own d-dim embedding. Numerical feature k is
       embedded as v_k = x_k * w_k + b_k; categorical as an
       ordinary embedding lookup.
    2. Standard Transformer encoder + [CLS] token for prediction.

Beats vanilla MLPs / TabTransformer / TabNet on most tabular
benchmarks (Gorishniy 2021 comparison).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def feature_tokenize_numeric(x_row, W, B):
    """Numerical feature-tokeniser: v_k = x_k * w_k + b_k for each feature."""
    F, d = W.shape
    return x_row[:, None] * W + B                                 # (F, d)


def feature_tokenize_cat(x_row_cat, embed_tables):
    """Categorical: lookup embedding per feature."""
    return np.array([embed_tables[k][int(x_row_cat[k])] for k in range(len(x_row_cat))])


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)


def self_attention(X, W_Q, W_K, W_V):
    Q = X @ W_Q; K = X @ W_K; V = X @ W_V
    d = Q.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d)) @ V


def ft_transformer(X, W_num, B_num, W_attn, W_head, cls_emb):
    """Compute prediction for each row of X."""
    preds = []
    for row in X:
        tokens = feature_tokenize_numeric(row, W_num, B_num)     # (F, d)
        tokens = np.vstack([cls_emb[None, :], tokens])           # prepend [CLS]
        z = self_attention(tokens, *W_attn) + tokens              # residual attn
        pred = z[0] @ W_head                                      # read [CLS]
        preds.append(pred)
    return np.array(preds)


if __name__ == "__main__":
    print("=== FT-Transformer (Gorishniy et al 2021 NeurIPS) ===\n")
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y = make_regression(n_samples=500, n_features=8, noise=1.0, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature-tokeniser embeddings (dim d=8)
    rng = np.random.default_rng(0)
    F, d = 8, 8
    W_num = rng.normal(scale=0.3, size=(F, d))
    B_num = rng.normal(scale=0.3, size=(F, d))
    cls_emb = rng.normal(scale=0.3, size=d)
    W_attn = tuple(rng.normal(scale=0.3, size=(d, d)) for _ in range(3))
    # Head: [CLS] -> scalar
    W_head = rng.normal(scale=0.3, size=d)
    # For a fair 'training' proxy, fit head via least squares on Xtr embeddings
    tr_embeds = np.array([
        (self_attention(np.vstack([cls_emb[None, :],
                                        feature_tokenize_numeric(row, W_num, B_num)]),
                          *W_attn) + np.vstack([cls_emb[None, :],
                                                     feature_tokenize_numeric(row, W_num, B_num)]))[0]
        for row in Xtr])
    W_head = np.linalg.lstsq(tr_embeds, ytr, rcond=None)[0]

    pred_ft = ft_transformer(Xte, W_num, B_num, W_attn, W_head, cls_emb)
    mse_ft = float(np.mean((pred_ft - yte) ** 2))

    # Baseline: linear regression
    lr = LinearRegression().fit(Xtr, ytr)
    mse_lr = float(np.mean((lr.predict(Xte) - yte) ** 2))

    print(f"  Task: regression, {len(Xtr)} train + {len(Xte)} test, F={F} features")
    print(f"  FT-Transformer ([CLS] + LS head) test MSE: {mse_ft:.2f}")
    print(f"  Linear regression baseline test MSE:       {mse_lr:.2f}")

    print(f"\n  Real FT-Transformer uses trained attention weights + FFN; here")
    print(f"  attention is random-init so only the LS head is fit.")

    print("\n--- library cross-check (yandex-research/rtdl; pytorch-tabular; tabnet) ---")

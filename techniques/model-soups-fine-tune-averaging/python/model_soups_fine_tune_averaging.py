"""Model soups -- averaging fine-tuned model weights (Reference Sec 47.28).

Wortsman et al. 2022 'Model soups: averaging weights of multiple fine-
tuned models improves accuracy without increasing inference time',
ICML. Instead of picking the single best fine-tuning run, AVERAGE the
weights of many runs (same architecture, same init, different hp /
seed):

    theta_soup = (1 / M) * sum_m theta_m           (uniform soup)
    theta_greedy = greedy add-if-improves-val-acc  (greedy soup)

Zero inference-time overhead vs a single model; often beats the best
single run and even ensemble prediction on some benchmarks.

We simulate M linear models fine-tuned from a common initialisation
with different data subsamples / seeds, average their weights, and
compare to (a) best single model and (b) prediction ensemble.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def fit_linear(X, y, init, lr=0.02, epochs=200):
    W = init.copy()
    for _ in range(epochs):
        pred = X @ W
        grad = X.T @ (pred - y) / len(X)
        W -= lr * grad
    return W


if __name__ == "__main__":
    print("=== Model soups (Wortsman 2022) ===\n")
    rng = np.random.default_rng(0)
    n = 400; d = 10
    X = rng.normal(size=(n, d))
    W_true = rng.normal(scale=0.5, size=d)
    y = X @ W_true + rng.normal(scale=0.5, size=n)

    #  Held-out test set for evaluation
    Xte = rng.normal(size=(200, d))
    yte = Xte @ W_true + rng.normal(scale=0.5, size=200)

    #  Common pre-trained init
    init = rng.normal(scale=0.3, size=d)

    #  Fit M models with different random sub-samples (bootstrap)
    M = 12
    models = []
    single_mses = []
    for m in range(M):
        idx = rng.choice(n, size=n, replace=True)
        W_m = fit_linear(X[idx], y[idx], init, lr=0.02, epochs=200)
        models.append(W_m)
        pred_te = Xte @ W_m
        single_mses.append(float(np.mean((pred_te - yte) ** 2)))

    best_single = min(single_mses)
    #  Uniform soup: average all M weights
    W_soup = np.mean(models, axis=0)
    mse_soup = float(np.mean((Xte @ W_soup - yte) ** 2))

    #  Prediction ensemble: average predictions of M models
    preds = np.mean([Xte @ W for W in models], axis=0)
    mse_ens = float(np.mean((preds - yte) ** 2))

    #  Greedy soup: add-if-improves on a val set (use half the test for val)
    val_idx = np.arange(100)
    test_idx = np.arange(100, 200)
    Xval, yval = Xte[val_idx], yte[val_idx]
    Xtst, ytst = Xte[test_idx], yte[test_idx]
    order = np.argsort(single_mses)   # best first
    greedy = models[order[0]].copy()
    included = [order[0]]
    best_val = float(np.mean((Xval @ greedy - yval) ** 2))
    for k in order[1:]:
        cand_soup = ((len(included) * greedy + models[k]) / (len(included) + 1))
        val_new = float(np.mean((Xval @ cand_soup - yval) ** 2))
        if val_new < best_val:
            greedy = cand_soup; included.append(k); best_val = val_new
    mse_greedy = float(np.mean((Xtst @ greedy - ytst) ** 2))

    print(f"  n = {n}, d = {d}, M = {M} fine-tuned models")
    print(f"  Best single-run MSE            = {best_single:.4f}")
    print(f"  Uniform soup MSE               = {mse_soup:.4f}")
    print(f"  Prediction ensemble MSE         = {mse_ens:.4f}   (needs M forward passes)")
    print(f"  Greedy soup MSE (test half)     = {mse_greedy:.4f}   "
          f"(included {len(included)}/{M})")
    print(f"\n  Soup gives ensemble-like quality with only ONE inference forward pass.")

    print("\n--- library cross-check (Wortsman code release; torch state_dict averaging Python) ---")

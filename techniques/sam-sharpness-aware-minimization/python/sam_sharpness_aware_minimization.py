"""Sharpness-Aware Minimization - SAM (Reference Sec 47.163).

Foret et al 2021 'Sharpness-Aware Minimization for Efficiently
Improving Generalization', ICLR. Instead of minimising the loss
at theta, SAM minimises the loss in a NEIGHBOURHOOD around theta:

    min_theta max_{||eps|| <= rho} L(theta + eps).

Two-step update:
    1. eps* = rho * grad L(theta) / || grad L(theta) ||   (ascent)
    2. theta_{t+1} = theta_t - lr * grad L(theta_t + eps*)  (descent)

Doubles compute per step, but flat minima -> improved test acc
(especially with limited data / label noise).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mlp_forward(params, X):
    W1, b1, W2, b2 = params
    h = np.maximum(X @ W1 + b1, 0)                              # ReLU hidden
    z = h @ W2 + b2                                              # scalar output
    return z.squeeze(-1), h


def logistic_loss_grad(params, X, y):
    """Cross-entropy loss + gradient for a 1-hidden-layer MLP (non-convex)."""
    W1, b1, W2, b2 = params
    z, h = mlp_forward(params, X)
    p = 1 / (1 + np.exp(-z))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    dz = (p - y) / len(X)
    dW2 = h.T @ dz[:, None]; db2 = dz.sum(keepdims=True)
    dh = dz[:, None] @ W2.T; dh = dh * (h > 0)
    dW1 = X.T @ dh; db1 = dh.sum(axis=0)
    return loss, (dW1, db1, dW2, db2)


def init_mlp(d, h, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(scale=0.5, size=(d, h)),
            np.zeros(h),
            rng.normal(scale=0.5, size=(h, 1)),
            np.zeros(1)]


def flat_params(params):
    return np.concatenate([p.ravel() for p in params])


def unflat_params(flat, shapes):
    out, i = [], 0
    for s in shapes:
        size = int(np.prod(s))
        out.append(flat[i:i + size].reshape(s))
        i += size
    return out


def grad_norm(grads):
    return float(np.sqrt(sum(float((g ** 2).sum()) for g in grads)))


def sgd(X, y, h=8, lr=0.05, n_iter=1500, seed=0):
    params = init_mlp(X.shape[1], h, seed=seed)
    for it in range(n_iter):
        _, grads = logistic_loss_grad(params, X, y)
        for i in range(len(params)):
            params[i] = params[i] - lr * grads[i]
    return params


def sam(X, y, h=8, lr=0.05, rho=0.1, n_iter=1500, seed=0):
    params = init_mlp(X.shape[1], h, seed=seed)
    for it in range(n_iter):
        _, g1 = logistic_loss_grad(params, X, y)
        gn = grad_norm(g1) + 1e-12
        params_pert = [p + rho * g / gn for p, g in zip(params, g1)]
        _, g2 = logistic_loss_grad(params_pert, X, y)
        for i in range(len(params)):
            params[i] = params[i] - lr * g2[i]
    return params


def flatness(params, X, y, rho, n_samples=50, seed=0):
    rng = np.random.default_rng(seed)
    L0, _ = logistic_loss_grad(params, X, y)
    max_dL = 0.0
    shapes = [p.shape for p in params]
    flat = flat_params(params)
    for _ in range(n_samples):
        eps = rng.normal(size=flat.shape); eps *= rho / (np.linalg.norm(eps) + 1e-12)
        pert = unflat_params(flat + eps, shapes)
        L, _ = logistic_loss_grad(pert, X, y)
        max_dL = max(max_dL, L - L0)
    return max_dL


if __name__ == "__main__":
    print("=== Sharpness-Aware Minimization - SAM (Foret et al 2021) ===\n")
    from sklearn.datasets import make_moons

    rng = np.random.default_rng(0)
    X, y = make_moons(n_samples=200, noise=0.35, random_state=0)
    # Add polynomial features + bias to make the problem overparametrised
    X = np.column_stack([X, X ** 2, X[:, 0] * X[:, 1], np.ones(len(X))])
    y = y.astype(float)
    # Add label noise to make sharpness matter
    flip = rng.uniform(size=len(y)) < 0.10
    y[flip] = 1 - y[flip]

    # Split
    perm = rng.permutation(len(X))
    tr, te = perm[:130], perm[130:]
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    def acc(params):
        z, _ = mlp_forward(params, Xte)
        return float(np.mean((z > 0) == (yte > 0.5)))

    p_sgd = sgd(Xtr, ytr, h=16, lr=0.1, n_iter=2000, seed=0)
    p_sam = sam(Xtr, ytr, h=16, lr=0.1, rho=0.3, n_iter=2000, seed=0)
    print(f"  200-sample moons with 10% label noise, 6-D poly features, MLP h=16")
    print(f"  SGD  test acc = {acc(p_sgd):.3f}")
    print(f"  SAM  test acc = {acc(p_sam):.3f}")
    print(f"  SGD  sharpness (max dL in rho=0.3 ball, 50 samples): {flatness(p_sgd, Xtr, ytr, 0.3):.4f}")
    print(f"  SAM  sharpness (max dL in rho=0.3 ball, 50 samples): {flatness(p_sam, Xtr, ytr, 0.3):.4f}")

    print("\n--- library cross-check (SAM-optimizer / pytorch impls; no R equivalent) ---")

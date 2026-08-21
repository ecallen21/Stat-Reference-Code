"""Dropout + BatchNorm from scratch (Reference §27.10).

DROPOUT (Srivastava et al. 2014):
    training: y = mask * x / (1 - p)   with mask ~ Bernoulli(1 - p)
    eval:     y = x                       (inverted-dropout scaling absorbed)

BATCH NORM (Ioffe-Szegedy 2015):
    training: normalise per feature over the mini-batch:
        mu_b = mean_b(x);  var_b = var_b(x)
        x_hat = (x - mu_b) / sqrt(var_b + eps)
        y = gamma * x_hat + beta
    eval:     use exponentially-moving-averaged mu and var from training.

Both regularise; BN also accelerates training by centring and standardising
activations layer-wise.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def dropout(x, p: float, training: bool, rng=None):
    if not training or p <= 0:
        return x, None
    if rng is None:
        rng = np.random.default_rng()
    mask = (rng.uniform(size=x.shape) > p).astype(float) / (1 - p)
    return x * mask, mask


class BatchNorm:
    def __init__(self, d, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(d); self.beta = np.zeros(d)
        self.mu_run = np.zeros(d); self.var_run = np.ones(d)
        self.momentum = momentum; self.eps = eps

    def forward(self, x, training: bool):
        if training:
            mu = x.mean(axis=0); var = x.var(axis=0)
            self.mu_run = self.momentum * self.mu_run + (1 - self.momentum) * mu
            self.var_run = self.momentum * self.var_run + (1 - self.momentum) * var
        else:
            mu, var = self.mu_run, self.var_run
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # DROPOUT — verify variance-preserving inverted scaling
    x = rng.normal(size=(1000, 4))
    y_drop, mask = dropout(x, p=0.5, training=True, rng=rng)
    y_eval, _ = dropout(x, p=0.5, training=False)
    print("=== Dropout ===")
    print(f"  input mean {x.mean():.3f}, var {x.var():.3f}")
    print(f"  train (p=0.5) mean {y_drop.mean():.3f}, var {y_drop.var():.3f}")
    print(f"    inverted scaling preserves E[y] = E[x] but inflates Var(y) ~ Var(x)/(1-p) — noise injection")
    print(f"  eval  (no-op) mean {y_eval.mean():.3f}, var {y_eval.var():.3f}")
    print(f"  fraction zeroed = {(mask == 0).mean():.3f}   (target = 0.5)")

    # BATCH NORM — verify per-feature zero-mean unit-variance during training
    bn = BatchNorm(d=4)
    x_big = rng.normal(loc=3.0, scale=5.0, size=(200, 4))
    for _ in range(50):
        y_bn = bn.forward(x_big, training=True)
    print("\n=== BatchNorm (after 50 training passes) ===")
    print(f"  input mean per feature: {np.round(x_big.mean(axis=0), 3).tolist()}")
    print(f"  input sd   per feature: {np.round(x_big.std(axis=0), 3).tolist()}")
    print(f"  bn output mean per feature: {np.round(y_bn.mean(axis=0), 3).tolist()}   "
          f"(gamma=1, beta=0 -> should be ~0)")
    print(f"  bn output sd   per feature: {np.round(y_bn.std(axis=0), 3).tolist()}   "
          f"(should be ~1)")
    print(f"  running mean : {np.round(bn.mu_run, 3).tolist()}")
    print(f"  running var  : {np.round(bn.var_run, 3).tolist()}")

    # eval-mode uses running stats
    x_test = rng.normal(loc=3.0, scale=5.0, size=(50, 4))
    y_eval = bn.forward(x_test, training=False)
    print(f"  eval-mode output mean per feature: {np.round(y_eval.mean(axis=0), 3).tolist()}")

    print("\n--- library cross-check (torch.nn.Dropout, torch.nn.BatchNorm1d) ---")
    try:
        import torch, torch.nn as nn
        drop = nn.Dropout(0.5)
        drop.train()
        y = drop(torch.tensor(x)).numpy()
        print(f"  torch dropout(p=0.5, train) var = {y.var():.3f}")
        bn_t = nn.BatchNorm1d(4)
        bn_t.train()
        for _ in range(50):
            _ = bn_t(torch.tensor(x_big, dtype=torch.float32))
        bn_t.eval()
        with torch.no_grad():
            y = bn_t(torch.tensor(x_test, dtype=torch.float32)).numpy()
        print(f"  torch BN eval-mode output mean = {np.round(y.mean(axis=0), 3).tolist()}")
    except ImportError:
        print("  (pytorch not installed)")

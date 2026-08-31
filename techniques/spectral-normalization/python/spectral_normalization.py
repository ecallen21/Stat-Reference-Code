"""Spectral normalization (Reference Ch 30 Robustness).

Miyato, Kataoka, Koyama & Yoshida (2018) "Spectral Normalization for
Generative Adversarial Networks."

Constrain each weight matrix W to have SPECTRAL NORM sigma(W) <= 1 (or
some cap) by DIVIDING W by its largest singular value:

  W_bar  =  W / sigma(W).

Applied per layer, this makes the network 1-Lipschitz -- small input
perturbations produce bounded output perturbations. Widely used in
GANs (stable training), Wasserstein-GP alternative, and modern
robustness pipelines (SNGP; distance-aware last layer).

POWER ITERATION for the top singular value (Miyato's O(1) trick):

  u_new = W v / |W v|
  v_new = W^T u / |W^T u|
  sigma  = u^T W v.

Only a few iterations per training step -- essentially free.

Here we demonstrate power-iteration + spectral normalisation on a random
matrix; then apply it to a small 3-layer MLP so the Lipschitz constant
is bounded to 1.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def power_iteration(W, u=None, n_iter=1):
    if u is None:
        u = np.random.default_rng(0).standard_normal(W.shape[0])
        u /= np.linalg.norm(u) + 1e-12
    for _ in range(n_iter):
        v = W.T @ u
        v /= np.linalg.norm(v) + 1e-12
        u = W @ v
        u /= np.linalg.norm(u) + 1e-12
    sigma = float(u @ (W @ v))
    return sigma, u, v


def spectral_normalize(W, u=None, n_iter=1, cap=1.0):
    sigma, u, v = power_iteration(W, u=u, n_iter=n_iter)
    W_bar = W * (cap / max(sigma, 1e-12))
    return W_bar, sigma, u


def _relu(x): return np.maximum(x, 0.0)


def build_mlp(rng, dims):
    return [{"W": rng.normal(0, np.sqrt(2 / dims[i]), (dims[i + 1], dims[i])),
              "u": rng.standard_normal(dims[i + 1])}
             for i in range(len(dims) - 1)]


def forward(net, x, sn=False, cap=1.0):
    h = x
    for i, layer in enumerate(net):
        W = layer["W"]
        if sn:
            W, sigma, u = spectral_normalize(W, u=layer["u"], n_iter=1, cap=cap)
            layer["u"] = u   # power iteration accumulates across steps in real training
        h = W @ h
        if i < len(net) - 1:
            h = _relu(h)
    return h


def lipschitz_probe(net, x0, n_probe=200, radius=1.0, sn=False, cap=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    y0 = forward(net, x0, sn=sn, cap=cap)
    max_ratio = 0.0
    for _ in range(n_probe):
        delta = rng.normal(0, 1, x0.shape)
        delta *= radius / (np.linalg.norm(delta) + 1e-12)
        y_new = forward(net, x0 + delta, sn=sn, cap=cap)
        ratio = np.linalg.norm(y_new - y0) / np.linalg.norm(delta)
        if ratio > max_ratio:
            max_ratio = ratio
    return max_ratio


if __name__ == "__main__":
    print("=== Spectral normalization (Miyato 2018) ===\n")
    rng = np.random.default_rng(0)
    # Sanity check: power iteration reproduces numpy SVD.
    W = rng.normal(0, 1, (5, 5))
    sig_pi, u, v = power_iteration(W, n_iter=100)
    sig_svd = np.linalg.svd(W, compute_uv=False)[0]
    print(f"  power-iter sigma = {sig_pi:.4f}   svd sigma = {sig_svd:.4f}"
          f"   |diff|={abs(sig_pi - sig_svd):.2e}\n")

    # 3-layer MLP; run 30 warmup power iterations on each layer's u for stability.
    dims = [10, 32, 32, 1]
    net = build_mlp(rng, dims)
    x0 = rng.normal(0, 1, dims[0])
    for _ in range(30):
        for layer in net:
            _, _, layer["u"] = spectral_normalize(layer["W"], u=layer["u"],
                                                     n_iter=1, cap=1.0)

    lip_raw = lipschitz_probe(net, x0, n_probe=400, sn=False, rng=rng)
    lip_sn  = lipschitz_probe(net, x0, n_probe=400, sn=True, cap=1.0, rng=rng)
    print(f"  empirical Lipschitz L2 (raw MLP)       : {lip_raw:.3f}")
    print(f"  empirical Lipschitz L2 (spectral-norm) : {lip_sn:.3f}   <- capped near 1.0\n")

    # Effect of cap
    for cap in (0.5, 1.0, 2.0, 5.0):
        lip = lipschitz_probe(net, x0, n_probe=200, sn=True, cap=cap, rng=rng)
        print(f"  cap={cap:.1f}   empirical Lipschitz={lip:.3f}")

    print("\n--- library cross-check (torch.nn.utils.parametrize.register_parametrization + SpectralNorm) ---")

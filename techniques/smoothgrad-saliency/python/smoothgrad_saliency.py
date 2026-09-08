"""SmoothGrad (Reference Sec 47.115).

Smilkov, Thorat, Kim, Viegas & Wattenberg 2017 'SmoothGrad:
removing noise by adding noise', arXiv:1706.03825. Averages raw
saliency maps over N Gaussian-perturbed copies of the input:

    S_hat(x) = (1/N) sum_{i=1}^{N}  |grad_x f(x + noise_i)|
    noise_i ~ N(0, sigma^2 I).

Denoises noisy gradient maps common on ReLU / max-pool networks;
sigma controls smoothness (larger sigma = smoother, less
class-specific).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def raw_gradient(x, W):
    """Gradient of a piecewise-linear-with-ReLU model f(x) = ReLU(W x)."""
    z = W @ x
    return (z > 0).astype(float)[:, None] * W          # ReLU passes gradient on active units


def scalar_output_grad(x, W, w_out):
    """f(x) = w_out . ReLU(W x);  grad_x f = W' diag(1{Wx>0}) w_out."""
    active = (W @ x > 0).astype(float)
    return W.T @ (active * w_out)


def smoothgrad(x, grad_fn, sigma, N, rng=None):
    rng = rng or np.random.default_rng(0)
    S = np.zeros_like(x)
    for _ in range(N):
        noise = rng.normal(0, sigma, size=x.shape)
        S += np.abs(grad_fn(x + noise))
    return S / N


if __name__ == "__main__":
    print("=== SmoothGrad (Smilkov et al 2017) ===\n")
    rng = np.random.default_rng(0)

    # Toy model:  f(x) = w_out . ReLU(W x)
    d = 20
    W = rng.normal(size=(30, d)) * 0.5
    w_out = rng.normal(size=30)
    # True 'signal' features: only 5 of 20
    idx_signal = [0, 3, 7, 12, 17]
    x = 0.5 * rng.normal(size=d)
    x[idx_signal] += 2.0

    raw = np.abs(scalar_output_grad(x, W, w_out))
    for sigma in [0.0, 0.1, 0.5, 1.0]:
        S = smoothgrad(x, lambda z: scalar_output_grad(z, W, w_out),
                        sigma=sigma, N=100, rng=rng)
        # Rank correlation between saliency and truth (indicator)
        truth = np.zeros(d); truth[idx_signal] = 1
        # top-5 recall
        top5 = np.argsort(-S)[:5]
        recall = float(len(set(top5.tolist()) & set(idx_signal)) / 5)
        print(f"  sigma = {sigma:.2f}  top-5 signal recall = {recall:.2f}   "
              f"S range = [{S.min():.3f}, {S.max():.3f}]")

    print("\n  Sigma = 0 is the raw gradient; larger sigma denoises noisy peaks.")
    print("  Too large sigma smears the signal across neighbours.")

    print("\n--- library cross-check (captum SmoothGrad / iNNvestigate Python) ---")

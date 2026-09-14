"""Straight-Through Estimator (Reference Sec 47.316).

Bengio, Leonard & Courville 2013 arXiv. Backpropagate through
a NON-DIFFERENTIABLE step (argmax, sign, binarisation) by using
the IDENTITY gradient in the backward pass:

    forward:  y = f(x)                        (e.g. argmax or sign)
    backward: dL/dx = dL/dy                   (as if f = identity)

Biased but LOW-VARIANCE; the workhorse for training quantised
networks, discrete-latent VAEs (VQ-VAE), and stochastic binary
networks.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def hard_binarise(x): return (x > 0).astype(float) * 2 - 1        # sign
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def ste_binary_forward(x): return hard_binarise(x)
def ste_binary_backward(dL_dy): return dL_dy                       # identity STE


def ste_binary_train_step(x, w, target, lr=0.1):
    """One SGD step through a binary weight via STE."""
    w_bin = ste_binary_forward(w)
    y = np.dot(w_bin, x)
    loss = (y - target) ** 2
    # Backward: dL/dw_bin = 2 (y - target) * x
    dL_dw_bin = 2 * (y - target) * x
    # STE: pass dL/dw_bin straight through to dL/dw
    dL_dw = ste_binary_backward(dL_dw_bin)
    return w - lr * dL_dw, float(loss)


if __name__ == "__main__":
    print("=== Straight-Through Estimator (Bengio et al 2013) ===\n")
    rng = np.random.default_rng(0)

    # Task: fit a binary-weight linear model to reproduce a target
    d = 6
    x = rng.standard_normal(d)
    w_true = np.random.default_rng(42).choice([-1, 1], size=d).astype(float)
    target = float(np.dot(w_true, x))
    print(f"  Target y = {target:.3f}   (true binary weights: {w_true.tolist()})\n")

    # Continuous latent w; effective binary weights via sign()
    w = rng.normal(0, 0.5, d)
    print(f"  Iter  loss  binarised weights = sign(w)")
    for it in range(30):
        w, loss = ste_binary_train_step(x, w, target, lr=0.05)
        if it % 5 == 0:
            print(f"  {it:>4}  {loss:>6.3f}  {ste_binary_forward(w).astype(int).tolist()}")

    print(f"\n  Final loss: {loss:.5f}")
    print(f"  Final binarised weights: {ste_binary_forward(w).astype(int).tolist()}")

    # Show gradient path
    print(f"\n  Gradient flow via STE:")
    print(f"    forward:  w -> sign(w) [step]")
    print(f"    backward: dL/dw = dL/d sign(w)  (skip the step)")
    print(f"    (Real chain rule would give dL/dw = 0 almost everywhere.)")
    print(f"    STE trades bias for a usable gradient signal.")

    print("\n--- library cross-check (torch.autograd.Function custom .backward = identity) ---")

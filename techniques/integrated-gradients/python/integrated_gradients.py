"""Integrated gradients (Reference Sec 47.30).

Sundararajan, Taly & Yan 2017 'Axiomatic attribution for deep
networks', ICML. Attribution method for differentiable models
satisfying the SENSITIVITY and IMPLEMENTATION INVARIANCE axioms:

    IG_i(x) = (x_i - x_i^base) * integral_{alpha=0..1} d f(x^base + alpha * (x - x^base)) / d x_i  dalpha

Approximated by discretising the path with M steps (trapezoidal /
left-Riemann) and averaging gradients.

Completeness: sum_i IG_i(x) = f(x) - f(x^base).
Baseline: usually zero vector, average sample, or blur.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def integrated_gradients(f, grad_f, x, x_base=None, steps=50):
    """f: scalar function; grad_f: vector gradient wrt x."""
    if x_base is None:
        x_base = np.zeros_like(x)
    alphas = np.linspace(0.0, 1.0, steps + 1)
    grads = np.array([grad_f(x_base + a * (x - x_base)) for a in alphas])
    avg_grad = (grads[:-1] + grads[1:]).mean(axis=0) / 2
    return (x - x_base) * avg_grad


if __name__ == "__main__":
    print("=== Integrated gradients (Sundararajan 2017) ===\n")
    #  Model: f(x) = tanh(x0) + 0.5 * x1^2 + 0.2 * x0 * x2
    def f(x):
        return float(np.tanh(x[0]) + 0.5 * x[1] ** 2 + 0.2 * x[0] * x[2])

    def grad_f(x):
        return np.array([1 - np.tanh(x[0]) ** 2 + 0.2 * x[2],
                          x[1],
                          0.2 * x[0]])

    x = np.array([1.0, 2.0, 3.0])
    x_base = np.array([0.0, 0.0, 0.0])

    ig = integrated_gradients(f, grad_f, x, x_base, steps=100)
    print(f"  x = {x.tolist()}, baseline = zeros")
    print(f"  f(x) - f(baseline) = {f(x) - f(x_base):+.4f}")
    print(f"  Integrated gradients: {ig.tolist()}")
    print(f"  Sum of IG (completeness check) = {ig.sum():+.4f}   (should match Delta f)")

    #  Compare with plain gradient at x (a bad attribution near saturation)
    plain_grad = grad_f(x)
    saliency_only = plain_grad * x
    print(f"\n  Plain 'gradient * input' saliency:  {saliency_only.tolist()}")
    print(f"    (does NOT satisfy completeness; over- or under-attributes at saturating activations)")

    print("\n--- library cross-check (captum.attr.IntegratedGradients / DeepLIFT Python) ---")

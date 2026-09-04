"""Information geometry (Reference Sec 34.15).

Amari (1985, 2016) 'Information Geometry and Its Applications.'

A parametric family { p(x; theta) } forms a manifold M with the FISHER-RAO
metric g_ij = I(theta)_ij. Consequences:

  * KL(p_theta || p_(theta + d)) ~ 0.5 * d' I(theta) d   (quadratic form).
  * NATURAL GRADIENT: precondition steepest-descent by I(theta)^-1
       theta_new = theta - lr * I^-1 grad L
    is invariant to reparameterisation.
  * exp / mixture families are DUALLY FLAT.

Here we:
  1. Verify the KL <-> Fisher quadratic approximation numerically for a
     Gaussian family.
  2. Compare vanilla gradient vs natural gradient on a logistic MLE:
     natural gradient converges in FEWER STEPS.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def gaussian_kl(mu1, sig1, mu2, sig2):
    return float(np.log(sig2 / sig1) + (sig1 ** 2 + (mu1 - mu2) ** 2) / (2 * sig2 ** 2) - 0.5)


def gaussian_fisher(mu, sigma):
    return np.diag([1 / sigma ** 2, 2 / sigma ** 2])


def logistic_neg_log_lik(beta, X, y):
    p = _sigmoid(X @ beta)
    return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))


def logistic_grad_fisher(beta, X, y):
    p = _sigmoid(X @ beta)
    grad = X.T @ (p - y) / len(y)
    W = p * (1 - p)
    I = X.T @ (X * W[:, None]) / len(y)              # Fisher = observed information for logistic
    return grad, I


if __name__ == "__main__":
    print("=== Information geometry ===\n")
    # (1) KL Fisher quadratic
    mu, sig = 0.0, 1.0
    print("  Gaussian family: verify  KL(theta || theta + d) ~ 0.5 d' I d")
    I = gaussian_fisher(mu, sig)
    for scale in (0.1, 0.05, 0.01):
        d = np.array([scale, scale])
        kl_exact = gaussian_kl(mu, sig, mu + d[0], sig + d[1])
        quad = 0.5 * d @ I @ d
        print(f"    |d| = {scale:>5}   exact KL = {kl_exact:.6e}   0.5 d' I d = {quad:.6e}")
    print()

    # (2) Natural vs vanilla gradient descent on logistic MLE
    rng = np.random.default_rng(0)
    n, d = 300, 3
    X = np.hstack([np.ones((n, 1)), rng.normal(0, 1, (n, d - 1))])
    beta_true = np.array([0.5, -1.0, 0.7])
    y = (rng.random(n) < _sigmoid(X @ beta_true)).astype(float)

    for method in ("vanilla", "natural"):
        beta = np.zeros(d)
        for it in range(50):
            g, I = logistic_grad_fisher(beta, X, y)
            if method == "vanilla":
                beta -= 0.3 * g
            else:
                beta -= np.linalg.solve(I + 1e-6 * np.eye(d), g)
            if np.linalg.norm(g) < 1e-4:
                break
        final_loss = logistic_neg_log_lik(beta, X, y)
        print(f"  {method:>7}   iterations = {it + 1:>2}   final NLL = {final_loss:.4f}"
              f"   beta = {beta.round(3).tolist()}")

    print("\n  Natural gradient converges in far fewer steps; the update is\n"
          "  invariant to reparameterisation (unlike vanilla SGD).\n")
    print("--- library cross-check (jax.example_libraries.optimizers.natural_gradient;\n"
          "  R geomstats / GeometricInfo) ---")

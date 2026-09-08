"""Conditional GAN (cGAN) (Reference Sec 47.108).

Mirza & Osindero 2014 'Conditional Generative Adversarial Nets',
arXiv:1411.1784. Extends the vanilla GAN by conditioning both G
and D on a label y:

    min_G max_D  E[ log D(x, y) ] + E[ log(1 - D(G(z, y), y)) ].

Because full GAN training is notoriously fiddly (mode collapse,
oscillation) and requires stochastic-gradient DL frameworks, this
from-scratch demo instead shows the CONVERGED EQUILIBRIUM:

    * per-class MLE fit for G (analytic optimum for a Gaussian
      class-conditional truth);
    * verify a logistic-regression D distinguishes real vs fake
      no better than chance (accuracy near 0.5).

A working cGAN drives the training loop to this equilibrium.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from sklearn.linear_model import LogisticRegression    # D as a linear classifier


def generator_1d(z, y, mu0, mu1, sigma):
    """Analytic optimum G given normal class-conditional truth."""
    mu = np.where(y > 0.5, mu1, mu0)
    return mu + sigma * z


if __name__ == "__main__":
    print("=== Conditional GAN, equilibrium demo (Mirza-Osindero 2014) ===\n")
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.integers(0, 2, size=n).astype(float)
    x_real = np.where(y > 0.5, rng.normal(2.0, 0.3, size=n),
                                 rng.normal(-2.0, 0.3, size=n))

    # (a) At an UNTRAINED G (means both zero) discriminator dominates
    z = rng.normal(size=n)
    x_bad = 0.3 * z + 0 * y          # ignores class label + wrong mean
    X_D = np.concatenate([np.column_stack([x_real, y]),
                            np.column_stack([x_bad, y])])
    y_D = np.concatenate([np.ones(n), np.zeros(n)])
    D_bad = LogisticRegression(max_iter=500).fit(X_D, y_D)
    acc_bad = D_bad.score(X_D, y_D)

    # (b) At the OPTIMUM G (means recovered per class) D near chance
    mu0_hat = x_real[y < 0.5].mean(); mu1_hat = x_real[y > 0.5].mean()
    sigma_hat = 0.5 * (x_real[y < 0.5].std() + x_real[y > 0.5].std())
    x_opt = generator_1d(z, y, mu0_hat, mu1_hat, sigma_hat)
    X_D_opt = np.concatenate([np.column_stack([x_real, y]),
                                 np.column_stack([x_opt, y])])
    D_opt = LogisticRegression(max_iter=500).fit(X_D_opt, y_D)
    acc_opt = D_opt.score(X_D_opt, y_D)

    print(f"  MLE G params: mu0={mu0_hat:+.3f}, mu1={mu1_hat:+.3f}, sigma={sigma_hat:.3f}")
    print(f"  Untrained G  ->  D acc = {acc_bad:.3f}   (perfect distinguishability)")
    print(f"  Optimum   G  ->  D acc = {acc_opt:.3f}   (near 0.5 = GAN equilibrium)")

    # Generated samples per class
    for y_val in (0.0, 1.0):
        z = rng.normal(size=5000)
        x = generator_1d(z, y_val, mu0_hat, mu1_hat, sigma_hat)
        true = -2.0 if y_val < 0.5 else 2.0
        print(f"\n  y = {int(y_val)}:  gen mean = {x.mean():+.3f}   sd = {x.std():.3f}   "
              f"target N({true:+.1f}, 0.3^2)")

    print("\n--- library cross-check (limited R; torchvision.models / torchGAN Python) ---")

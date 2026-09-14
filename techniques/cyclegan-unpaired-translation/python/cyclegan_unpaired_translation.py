"""CycleGAN - Unpaired Image Translation (Ref Sec 47.287).

Zhu, Park, Isola & Efros 2017 ICCV. Learn a mapping G: X -> Y
between two DOMAINS without paired examples using a CYCLE
CONSISTENCY loss:

    L_GAN(G, D_Y) + L_GAN(F, D_X)                     (adversarial in both directions)
    + lambda * (||F(G(x)) - x||_1 + ||G(F(y)) - y||_1) (cycle consistency)

Two generators G, F and two discriminators D_X, D_Y are trained
jointly. Enables horse<->zebra, photo<->painting, summer<->winter
style transfer without paired training data.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_G(x, theta): return x + theta                             # simple shift generator
def toy_F(y, theta): return y - theta


def gan_loss_toy(D_out_real, D_out_fake):
    return -(np.log(np.clip(D_out_real, 1e-6, 1)) + np.log(np.clip(1 - D_out_fake, 1e-6, 1))).mean()


def cycle_consistency(x, y, theta_G, theta_F):
    xhat = toy_F(toy_G(x, theta_G), theta_F)
    yhat = toy_G(toy_F(y, theta_F), theta_G)
    return float(np.mean(np.abs(xhat - x)) + np.mean(np.abs(yhat - y)))


if __name__ == "__main__":
    print("=== CycleGAN (Zhu et al 2017 ICCV) ===\n")
    rng = np.random.default_rng(0)

    # Toy domains: X ~ N(0, 1), Y ~ N(2, 1) (learn shift +2)
    X = rng.normal(0, 1, (500, 1))
    Y = rng.normal(2, 1, (500, 1))

    # Naive shift estimator: means
    theta_direct = float(Y.mean() - X.mean())
    print(f"  Ground-truth domain shift = 2.0, empirical (means): {theta_direct:.3f}\n")

    # Simulated cycle-consistency loss surface
    print(f"  theta   cycle-loss  L1(x, F(G(x)))")
    for theta in [-1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0]:
        c = cycle_consistency(X, Y, theta_G=theta, theta_F=-theta)
        print(f"  {theta:>4.1f}   {c:>10.4f}")
    print(f"\n  Cycle loss is zero whenever theta_F = -theta_G, INDEPENDENT of theta.")
    print(f"  Cycle consistency alone does NOT identify the true shift -")
    print(f"  the adversarial loss on distribution matching is what pins theta = +2.\n")

    # Adversarial "distribution matching" surrogate: match sample mean of G(X) to Y
    print(f"  Adversarial surrogate (match means): argmin (E[G(X)] - E[Y])^2")
    for theta in [-1.0, 0.0, 1.0, 1.5, 2.0, 2.5, 3.0]:
        adv = (toy_G(X, theta).mean() - Y.mean()) ** 2
        print(f"    theta = {theta:>4.1f}   adv loss = {adv:.4f}")

    print("\n  Combining cycle consistency (uniqueness / structure preservation)")
    print("  with adversarial matching (distribution) is CycleGAN's key insight.")

    print("\n--- library cross-check (junyanz/pytorch-CycleGAN-and-pix2pix; tf-gan) ---")

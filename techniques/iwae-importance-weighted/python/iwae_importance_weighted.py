"""IWAE - Importance Weighted Autoencoder (Ref Sec 47.289).

Burda, Grosse & Salakhutdinov 2016 ICLR. Tighter ELBO for VAEs
via K importance samples:

    log p(x) >= L_K = E[log (1/K) sum_k w_k]
        w_k = p(x, z_k) / q(z_k | x),  z_k ~ q(z | x)

L_K increases monotonically with K, and L_1 recovers the plain
ELBO. Trades compute for a tighter bound; important for
posteriors that a factorised q(z|x) misfits.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def toy_log_p(x, z):
    """log p(x, z) = log N(x | z, sigma) + log N(z | 0, 1). 1-D toy."""
    log_p_x_z = -0.5 * (x - z) ** 2                              # sigma=1
    log_p_z   = -0.5 * z ** 2
    return log_p_x_z + log_p_z


def toy_log_q(z, mu, log_var):
    """log q(z | x) - Gaussian."""
    return -0.5 * ((z - mu) ** 2 / np.exp(log_var) + log_var)


def iwae_bound(x, mu, log_var, K, rng):
    """Estimate log p(x) via K-sample IS. Returns bound + variance."""
    z = mu + np.exp(0.5 * log_var) * rng.standard_normal(K)
    log_w = toy_log_p(x, z) - toy_log_q(z, mu, log_var)
    log_bound = float(np.log(np.mean(np.exp(log_w - log_w.max()))) + log_w.max())
    return log_bound, float(np.var(log_w))


if __name__ == "__main__":
    print("=== IWAE - Importance Weighted Autoencoder (Burda et al 2016) ===\n")
    rng = np.random.default_rng(0)

    # Toy: observed x = 1.5. Posterior of z is N(x/2, 1/2). Use a MIS-SPECIFIED
    # variational family q(z|x) = N(0.5, log_var=0) to show the effect of tightening K.
    x = 1.5
    mu, log_var = 0.5, 0.0

    true_log_p = -0.5 * np.log(2 * np.pi * 2) - x ** 2 / 4       # exact log p(x) = log N(x | 0, 2)
    print(f"  True log p(x) at x={x}: {true_log_p:.4f}\n")

    print(f"  IWAE bound vs K (mis-specified q):")
    print(f"  {'K':>5}   bound (mean of 50 reps)   var(log_w)   gap to truth")
    for K in [1, 2, 5, 20, 100, 500]:
        bounds = [iwae_bound(x, mu, log_var, K, rng)[0] for _ in range(50)]
        var_lw = iwae_bound(x, mu, log_var, K, rng)[1]
        b_mean = float(np.mean(bounds))
        print(f"  {K:>5}   {b_mean:>+.4f}   {var_lw:>10.3f}   {true_log_p - b_mean:.4f}")

    print(f"\n  IWAE bound TIGHTENS monotonically as K increases (Burda thm 1).")
    print(f"  K = 1 recovers plain ELBO; K -> inf recovers log p(x).")
    print(f"  Cost: K times the compute; benefit: better posterior when q misfits.")

    print("\n--- library cross-check (Pyro IWAE loss; TFP tfp.vi.monte_carlo_variational_loss(num_samples=K)) ---")

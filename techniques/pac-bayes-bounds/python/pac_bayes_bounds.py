"""PAC-Bayes bounds (Reference Sec 46.15).

McAllester 1999 'PAC-Bayesian model averaging', COLT; Catoni 2007
'PAC-Bayesian supervised classification'. Generalisation bounds for
randomised (posterior) hypotheses Q, controlled by KL divergence
from a data-independent prior P:

    With probability >= 1 - delta over the n iid sample S:

        E_{h ~ Q} L(h) <= E_{h ~ Q} L_hat_n(h)
                          + sqrt( (KL(Q || P) + log(2 sqrt(n) / delta)) / (2 n) )

Tighter forms:
    * Catoni:    slightly better constants
    * Seeger:    binary KL inversion (best-in-class for classification)
    * Maurer:    bounded losses, faster rates via variance

We estimate empirical (KL + log) / n on a toy linear-classifier
family with Gaussian prior P and Q both centred on lambda-shifted
means. This shows the bound as a function of prior/posterior shift.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def kl_gauss(mu_q, sigma_q, mu_p, sigma_p):
    """KL(N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)) for scalar Gaussians."""
    return (np.log(sigma_p / sigma_q)
            + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2)
            - 0.5)


def mcallester_bound(emp_loss, kl, n, delta=0.05):
    return emp_loss + np.sqrt((kl + np.log(2 * np.sqrt(n) / delta)) / (2 * n))


def catoni_bound(emp_loss, kl, n, delta=0.05, C=1.0):
    """Catoni-style bound (simplified): sqrt((KL + log(1/delta)) / (2 n))."""
    return emp_loss + np.sqrt((kl + np.log(1 / delta)) / (2 * n)) * C


if __name__ == "__main__":
    print("=== PAC-Bayes bounds (McAllester / Catoni) ===\n")
    rng = np.random.default_rng(0)
    n = 500

    #  Toy loss for a single-parameter classifier
    #    L(h) = P(y != sign(x - h)) with x ~ Uniform(-1, 1), y = sign(x)
    #  Optimal h = 0, loss 0.
    delta = 0.05

    print(f"  n = {n}, delta = {delta}\n")
    print(f"  {'Q shift mu_q':>15s}  {'KL(Q || P)':>12s}  {'emp loss':>10s}  "
          f"{'McAllester':>12s}  {'Catoni':>10s}")
    #  Prior P = N(0, 0.5^2). Vary posterior mean and evaluate the bound.
    mu_p, sigma_p = 0.0, 0.5
    sigma_q = 0.1
    for mu_q in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        #  Simulate empirical loss for a hypothesis at mu_q (single-sample proxy)
        x = rng.uniform(-1, 1, size=n)
        y = np.sign(x)
        emp = float(np.mean(y != np.sign(x - mu_q)))
        kl = kl_gauss(mu_q, sigma_q, mu_p, sigma_p)
        mB = mcallester_bound(emp, kl, n, delta)
        cB = catoni_bound(emp, kl, n, delta)
        print(f"  {mu_q:>15.2f}  {kl:>12.4f}  {emp:>10.4f}  {mB:>12.4f}  {cB:>10.4f}")

    print("\n  Interpretation: bounds are TIGHT when the posterior is close to the prior")
    print("  (KL small) and empirical loss is small. Very shifted posteriors (mu_q = 1.0)")
    print("  incur a big KL penalty in exchange for lower empirical loss -- may or may")
    print("  not lower the bound overall.")

    print("\n--- library cross-check (no dedicated package; from-scratch) ---")

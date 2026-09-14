"""Metropolis-adjusted Langevin Algorithm (Roberts & Tweedie 1996).

MALA proposes:

    y = x + tau * grad log pi(x) + sqrt(2 tau) * eta,  eta ~ N(0, I)

and accepts with Metropolis-Hastings probability

    alpha = min(1, pi(y) q(x|y) / (pi(x) q(y|x)))

Combines the fast mixing of Langevin dynamics with the
correctness of Metropolis-Hastings. Discretisation bias from
plain unadjusted Langevin is eliminated.

Target rate: ~ 57.4 % (Roberts-Rosenthal 1998 optimal for MALA).
"""

import numpy as np    # arrays + random


def mala(log_pi, grad_log_pi, x0, tau, n_iter, seed=0):
    rng = np.random.default_rng(seed)
    x = np.array(x0, dtype=float)
    d = len(x)
    samples = np.empty((n_iter, d))
    accept = 0
    lp_x = log_pi(x)
    g_x = grad_log_pi(x)
    for t in range(n_iter):
        eta = rng.standard_normal(d)
        y = x + tau * g_x + np.sqrt(2 * tau) * eta
        lp_y = log_pi(y)
        g_y = grad_log_pi(y)
        # log q(x | y) - log q(y | x)
        log_q_xy = -np.sum((x - y - tau * g_y) ** 2) / (4 * tau)
        log_q_yx = -np.sum((y - x - tau * g_x) ** 2) / (4 * tau)
        log_alpha = lp_y - lp_x + log_q_xy - log_q_yx
        if np.log(rng.uniform()) < log_alpha:
            x, lp_x, g_x = y, lp_y, g_y
            accept = accept + 1
        samples[t] = x
    return samples, accept / n_iter


def demo():
    print("=== MALA — Metropolis-Adjusted Langevin (Roberts-Tweedie 1996) ===")
    print("Target: multivariate Normal N(0, Sigma) with correlated Sigma, d=5")

    rng_setup = np.random.default_rng(0)
    d = 5
    A = rng_setup.standard_normal((d, d))
    Sigma = A @ A.T + np.eye(d)
    Sigma_inv = np.linalg.inv(Sigma)

    def log_pi(x):
        return -0.5 * x @ Sigma_inv @ x

    def grad_log_pi(x):
        return -Sigma_inv @ x

    for tau in [0.2, 0.5, 1.0, 2.0]:
        samples, rate = mala(log_pi, grad_log_pi, x0=np.zeros(d),
                             tau=tau, n_iter=20000, seed=1)
        samples = samples[2000:]
        emp_cov = np.cov(samples.T)
        err = np.linalg.norm(emp_cov - Sigma, "fro") / np.linalg.norm(Sigma, "fro")
        print(f"  tau = {tau:.3f}: accept rate = {rate:.3f}, "
              f"cov error = {err:.3f}   (target rate ~ 0.574)")


if __name__ == "__main__":
    demo()

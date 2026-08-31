"""SWAG — SWA-Gaussian (Reference Ch 29 Uncertainty Quantification).

Maddox, Izmailov, Garipov, Vetrov & Wilson (2019) "A Simple Baseline for
Bayesian Uncertainty in Deep Learning."

Take a trained network to a low-loss basin, then run K extra SGD steps with
a moderately-high constant learning rate and collect the iterates
theta_1..theta_K. Fit a Gaussian to those iterates:

  mu_SWA  = mean(theta_k)
  Sigma   = 0.5 * (Sigma_diag + Sigma_lowrank)
     Sigma_diag    = diag(mean(theta_k^2) - mu_SWA^2)
     Sigma_lowrank = D D^T / (K - 1),    D columns = theta_k - mu_SWA_running

Sample from N(mu_SWA, Sigma) at test time for predictive uncertainty. The
low-rank + diagonal factorisation costs only 2K x P memory (P = # weights).

Here we implement SWAG in the from-scratch setting: linear regression with
SGD, collect iterates, form the SWA-Gaussian, sample K posterior samples
and produce a predictive band.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sgd_iterates(x, y, lr=5e-3, epochs=200, K=30, seed=0):
    """Fit y ~ x @ theta with SGD; collect iterates every 'stride' epochs."""
    rng = np.random.default_rng(seed)
    n, d = x.shape
    theta = rng.normal(0, 0.5, d)
    # burn-in with decreasing lr
    for e in range(epochs):
        idx = rng.integers(0, n, size=min(16, n))
        xb, yb = x[idx], y[idx]
        g = 2 * xb.T @ (xb @ theta - yb) / len(xb)
        theta -= lr * g / (1 + e / 100)
    # collect K iterates with a fixed constant lr
    thetas = []
    stride = 10
    for e in range(K * stride):
        idx = rng.integers(0, n, size=min(16, n))
        xb, yb = x[idx], y[idx]
        g = 2 * xb.T @ (xb @ theta - yb) / len(xb)
        theta = theta - lr * g
        if (e + 1) % stride == 0:
            thetas.append(theta.copy())
    return np.array(thetas)  # (K, d)


def fit_swag(thetas):
    K, d = thetas.shape
    mu = thetas.mean(axis=0)
    sq = (thetas ** 2).mean(axis=0)
    sigma_diag = np.clip(sq - mu ** 2, 1e-12, None)
    # low-rank deviation matrix: theta_k - running-mean up to k
    D = np.zeros((d, K))
    run_mean = np.zeros(d)
    for k in range(K):
        run_mean = (run_mean * k + thetas[k]) / (k + 1)
        D[:, k] = thetas[k] - run_mean
    return {"mu": mu, "sigma_diag": sigma_diag, "D": D, "K": K}


def swag_sample(swag, rng, n_samples=100, scale=0.5):
    d = swag["mu"].size
    K = swag["K"]
    out = np.zeros((n_samples, d))
    for s in range(n_samples):
        z1 = rng.standard_normal(d)
        z2 = rng.standard_normal(K)
        diag_part = np.sqrt(swag["sigma_diag"]) * z1
        low_rank = swag["D"] @ z2 / np.sqrt(2 * (K - 1))
        out[s] = swag["mu"] + scale * (diag_part / np.sqrt(2) + low_rank)
    return out


if __name__ == "__main__":
    print("=== SWAG (Maddox 2019) ===\n")
    rng = np.random.default_rng(0)
    # noisy sinusoid, basis functions [1, x, x^2, x^3, x^4]
    x = rng.uniform(-2, 2, 60)
    y = np.sin(1.5 * x) + rng.normal(0, 0.2, 60)
    def basis(z):
        return np.stack([np.ones_like(z), z, z ** 2, z ** 3, z ** 4], axis=1)
    X = basis(x)
    thetas = sgd_iterates(X, y, K=40, seed=0)
    swag = fit_swag(thetas)
    print("  SWA mean weights (mu):", np.round(swag["mu"], 3))
    print("  diag sd:              ", np.round(np.sqrt(swag["sigma_diag"]), 3))

    # predictive samples
    x_te = np.linspace(-3, 3, 21)
    Xte = basis(x_te)
    posts = swag_sample(swag, rng, n_samples=300)   # (S, d)
    y_samples = Xte @ posts.T                        # (n_te, S)
    mu_pred = y_samples.mean(axis=1)
    sd_pred = y_samples.std(axis=1)

    print(f"\n  {'x':>6}  {'mu':>7}  {'sd':>6}  region")
    for i, xv in enumerate(x_te):
        region = "in " if -2 <= xv <= 2 else "out"
        print(f"  {xv:>6.2f}  {mu_pred[i]:>7.3f}  {sd_pred[i]:>6.3f}  {region}")

    in_mask = (x_te >= -2) & (x_te <= 2)
    ratio = sd_pred[~in_mask].mean() / sd_pred[in_mask].mean()
    print(f"\n  predictive sd ratio (out/in): {ratio:.2f}x   <- SWAG posterior widens on OOD.\n")

    print("--- library cross-check (torchcontrib.optim.SWA + swag.SWAG; pyro SWAG) ---")

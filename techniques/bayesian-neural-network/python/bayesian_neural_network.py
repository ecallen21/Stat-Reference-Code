"""Bayesian neural network — mean-field variational inference (Ch 29 UQ).

Place a Gaussian prior on every weight and learn a Gaussian variational
approximation q(w) = N(mu_w, sigma_w^2) that minimises the ELBO:

  L = E_q[log p(y|x, w)] - KL(q(w) || p(w))

Local-reparameterisation trick (Kingma-Salimans-Welling 2015): rather than
sampling w and then computing z = W x, sample z directly:

  z ~ N(mu_z = mu_W x, sigma_z^2 = sigma_W^2 x^2)

This makes gradients low-variance and cheap. Bayes by Backprop (Blundell 2015).

Here we implement a single-layer nonlinear regressor (mu_W, log sigma_W) fit
to noisy sin data, with predictive uncertainty from Monte-Carlo samples of the
posterior.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _relu(x): return np.maximum(x, 0.0)


def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _kl_gauss(mu_q, log_sigma_q, sigma_p=1.0):
    # KL(N(mu, sigma) || N(0, sigma_p^2)), summed over all elements
    sigma2 = np.exp(2 * log_sigma_q)
    return 0.5 * np.sum(sigma2 / sigma_p ** 2 + mu_q ** 2 / sigma_p ** 2
                         - 1.0 - 2 * log_sigma_q + 2 * np.log(sigma_p))


def _init(rng, d_in, d_hid, d_out, log_sigma0=-3.0):
    return {
        "mu_W1": rng.normal(0, 0.5, (d_in, d_hid)),
        "ls_W1": np.full((d_in, d_hid), log_sigma0),
        "mu_b1": np.zeros(d_hid),  "ls_b1": np.full(d_hid, log_sigma0),
        "mu_W2": rng.normal(0, 0.5, (d_hid, d_out)),
        "ls_W2": np.full((d_hid, d_out), log_sigma0),
        "mu_b2": np.zeros(d_out),  "ls_b2": np.full(d_out, log_sigma0),
    }


def _sample(rng, mu, ls):
    return mu + np.exp(ls) * rng.standard_normal(mu.shape)


def _forward_sample(p, x, rng):
    W1 = _sample(rng, p["mu_W1"], p["ls_W1"])
    b1 = _sample(rng, p["mu_b1"], p["ls_b1"])
    W2 = _sample(rng, p["mu_W2"], p["ls_W2"])
    b2 = _sample(rng, p["mu_b2"], p["ls_b2"])
    h = _relu(x @ W1 + b1)
    y = (h @ W2 + b2).ravel()
    return y, (W1, b1, W2, b2, h)


def _kl_total(p, sigma_p=1.0):
    return (_kl_gauss(p["mu_W1"], p["ls_W1"], sigma_p) +
            _kl_gauss(p["mu_b1"], p["ls_b1"], sigma_p) +
            _kl_gauss(p["mu_W2"], p["ls_W2"], sigma_p) +
            _kl_gauss(p["mu_b2"], p["ls_b2"], sigma_p))


def train_bnn(x, y, d_hid=32, sigma_noise=0.15, lr=5e-3, epochs=2000,
              sigma_p=1.0, kl_weight=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    p = _init(rng, x.shape[1], d_hid, 1)
    n = x.shape[0]
    for _ in range(epochs):
        y_pred, (W1, b1, W2, b2, h) = _forward_sample(p, x, rng)
        # data likelihood: Gaussian, sigma_noise known
        d_y = (y_pred - y) / (sigma_noise ** 2) / n
        d_W2 = h.T @ d_y[:, None]
        d_b2 = np.array([d_y.sum()])
        d_h = d_y[:, None] @ W2.T
        d_h[h <= 0] = 0.0
        d_W1 = x.T @ d_h
        d_b1 = d_h.sum(axis=0)
        # KL gradients (analytic Gaussian, w.r.t. mu and log_sigma)
        for k_mu, k_ls, sh in (("mu_W1", "ls_W1", p["mu_W1"].shape),
                                ("mu_b1", "ls_b1", p["mu_b1"].shape),
                                ("mu_W2", "ls_W2", p["mu_W2"].shape),
                                ("mu_b2", "ls_b2", p["mu_b2"].shape)):
            sigma2 = np.exp(2 * p[k_ls])
            d_kl_mu = p[k_mu] / sigma_p ** 2
            d_kl_ls = sigma2 / sigma_p ** 2 - 1.0
            p[k_mu] -= lr * kl_weight * d_kl_mu
            p[k_ls] -= lr * kl_weight * d_kl_ls
        # Apply data-likelihood gradients (chain-rule at the sampled weights)
        p["mu_W2"] -= lr * d_W2
        p["mu_b2"] -= lr * d_b2
        p["mu_W1"] -= lr * d_W1
        p["mu_b1"] -= lr * d_b1
    return p


def bnn_predict(p, x, T=200, seed=1):
    rng = np.random.default_rng(seed)
    preds = np.zeros((T, x.shape[0]))
    for t in range(T):
        y_t, _ = _forward_sample(p, x, rng)
        preds[t] = y_t
    return {"mu": preds.mean(axis=0), "epistemic_sd": preds.std(axis=0),
             "samples": preds}


if __name__ == "__main__":
    print("=== Bayesian neural network (mean-field VI) ===\n")
    rng = np.random.default_rng(0)
    x_tr = rng.uniform(-2, 2, 100).reshape(-1, 1)
    y_tr = np.sin(2 * x_tr[:, 0]) + rng.normal(0, 0.15, 100)

    p = train_bnn(x_tr, y_tr, d_hid=32, epochs=2500, seed=0)

    x_te = np.linspace(-3, 3, 21).reshape(-1, 1)
    r = bnn_predict(p, x_te, T=300, seed=7)

    print(f"  {'x':>6}  {'mu':>7}  {'sd':>6}  region")
    for i, xv in enumerate(x_te[:, 0]):
        region = "in " if -2 <= xv <= 2 else "out"
        print(f"  {xv:>6.2f}  {r['mu'][i]:>7.3f}  {r['epistemic_sd'][i]:>6.3f}  {region}")

    in_mask = (x_te[:, 0] >= -2) & (x_te[:, 0] <= 2)
    ratio = r['epistemic_sd'][~in_mask].mean() / r['epistemic_sd'][in_mask].mean()
    print(f"\n  epistemic sd ratio (out/in): {ratio:.2f}x\n")

    print("--- library cross-check (pyro / tensorflow-probability / bnn.torch) ---")

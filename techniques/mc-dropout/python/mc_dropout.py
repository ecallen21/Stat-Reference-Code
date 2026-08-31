"""MC dropout (Reference Ch 29 Uncertainty Quantification).

Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation: Representing
Model Uncertainty in Deep Learning."

Standard dropout is applied only at training time; MC dropout keeps dropout
ON at INFERENCE time and averages T stochastic forward passes:

  mu(x)  = (1/T) sum_t f_hat_t(x)
  var(x) = tau^{-1} + (1/T) sum_t f_hat_t(x)^2 - mu(x)^2

Gal showed that dropout applied before every weight layer is equivalent to a
variational approximation to a deep GP posterior; the sample-variance of the
predictions is an estimator of predictive uncertainty. Cheap, drop-in, no
extra parameters vs training the network normally.

Here we implement dropout MLP with train-time backprop, then run MC dropout
at inference to get (mu, var).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _relu(x): return np.maximum(x, 0.0)


def _init(rng, d_in, d_hid, d_out):
    return {
        "W1": rng.normal(0, np.sqrt(2.0 / d_in), (d_in, d_hid)),
        "b1": np.zeros(d_hid),
        "W2": rng.normal(0, np.sqrt(2.0 / d_hid), (d_hid, d_hid)),
        "b2": np.zeros(d_hid),
        "W3": rng.normal(0, np.sqrt(2.0 / d_hid), (d_hid, d_out)),
        "b3": np.zeros(d_out),
    }


def _forward(p, x, rng=None, p_drop=0.0, train=False):
    # Two hidden layers, dropout before each weight (a la Gal 2016).
    h1_pre = x @ p["W1"] + p["b1"]
    h1 = _relu(h1_pre)
    m1 = None
    if p_drop > 0 and rng is not None:
        m1 = (rng.random(h1.shape) > p_drop).astype(float) / (1.0 - p_drop)
        h1 = h1 * m1
    h2_pre = h1 @ p["W2"] + p["b2"]
    h2 = _relu(h2_pre)
    m2 = None
    if p_drop > 0 and rng is not None:
        m2 = (rng.random(h2.shape) > p_drop).astype(float) / (1.0 - p_drop)
        h2 = h2 * m2
    y = (h2 @ p["W3"] + p["b3"]).ravel()
    cache = (x, h1_pre, h1, m1, h2_pre, h2, m2)
    return y, cache


def train(x, y, p_drop=0.1, d_hid=64, lr=1e-2, epochs=1500, seed=0):
    rng = np.random.default_rng(seed)
    p = _init(rng, x.shape[1], d_hid, 1)
    n = x.shape[0]
    for _ in range(epochs):
        yhat, cache = _forward(p, x, rng=rng, p_drop=p_drop, train=True)
        _, h1_pre, h1, m1, h2_pre, h2, m2 = cache
        d_y = 2 * (yhat - y) / n
        d_W3 = h2.T @ d_y[:, None]
        d_b3 = np.array([d_y.sum()])
        d_h2 = d_y[:, None] @ p["W3"].T
        if m2 is not None: d_h2 = d_h2 * m2
        d_h2[h2_pre <= 0] = 0.0
        d_W2 = h1.T @ d_h2
        d_b2 = d_h2.sum(axis=0)
        d_h1 = d_h2 @ p["W2"].T
        if m1 is not None: d_h1 = d_h1 * m1
        d_h1[h1_pre <= 0] = 0.0
        d_W1 = x.T @ d_h1
        d_b1 = d_h1.sum(axis=0)
        for k, g in (("W1", d_W1), ("b1", d_b1), ("W2", d_W2), ("b2", d_b2),
                      ("W3", d_W3.ravel().reshape(p["W3"].shape)), ("b3", d_b3)):
            p[k] -= lr * g
    return p


def mc_dropout_predict(p, x, T=100, p_drop=0.1, tau=1.0, seed=1):
    rng = np.random.default_rng(seed)
    preds = np.zeros((T, x.shape[0]))
    for t in range(T):
        yt, _ = _forward(p, x, rng=rng, p_drop=p_drop, train=False)
        preds[t] = yt
    mu = preds.mean(axis=0)
    var = preds.var(axis=0) + 1.0 / tau
    return {"mu": mu, "var": var, "epistemic": preds.var(axis=0),
             "aleatoric_prior": 1.0 / tau, "samples": preds}


if __name__ == "__main__":
    print("=== MC dropout (Gal & Ghahramani 2016) ===\n")
    rng = np.random.default_rng(0)
    x_tr = rng.uniform(-2, 2, 120).reshape(-1, 1)
    y_tr = np.sin(2 * x_tr[:, 0]) + rng.normal(0, 0.15, 120)

    p_drop = 0.10
    p = train(x_tr, y_tr, p_drop=p_drop, d_hid=64, epochs=1500, seed=0)

    x_te = np.linspace(-3, 3, 21).reshape(-1, 1)
    r = mc_dropout_predict(p, x_te, T=200, p_drop=p_drop, tau=25.0, seed=7)

    print(f"  {'x':>6}  {'mu':>7}  {'sd':>6}  {'epist_sd':>9}  region")
    for i, xv in enumerate(x_te[:, 0]):
        region = "in " if -2 <= xv <= 2 else "out"
        print(f"  {xv:>6.2f}  {r['mu'][i]:>7.3f}  {np.sqrt(r['var'][i]):>6.3f}"
              f"  {np.sqrt(r['epistemic'][i]):>9.3f}  {region}")

    in_mask = (x_te[:, 0] >= -2) & (x_te[:, 0] <= 2)
    ratio = np.sqrt(r['epistemic'][~in_mask]).mean() / np.sqrt(r['epistemic'][in_mask]).mean()
    print(f"\n  epistemic sd ratio (out/in): {ratio:.2f}x  <- should be > 1\n")

    print("--- library cross-check (torch nn.Dropout with model.train() at inference) ---")

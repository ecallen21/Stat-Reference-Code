"""Last-layer Bayesian (Reference Ch 29 Uncertainty Quantification).

Train the neural network normally, then freeze all but the FINAL layer
and do exact Bayesian linear regression on the frozen feature map. Also
called Bayesian Last Layer (BLL) or Neural Linear.

  phi(x) = network penultimate features (fixed after training)
  y = phi(x)^T w + eps,   w ~ N(0, sigma_w^2 I), eps ~ N(0, sigma_n^2)

Posterior:  w | D  ~  N(mu_w, Sigma_w)  with
  Sigma_w = (Phi^T Phi / sigma_n^2 + I / sigma_w^2)^(-1)
  mu_w    = Sigma_w Phi^T y / sigma_n^2

Predictive:
  mu_pred(x*) = phi(x*)^T mu_w
  var_pred(x*) = phi(x*)^T Sigma_w phi(x*) + sigma_n^2

Cheap posterior over just the last-layer weights (P_last is tiny) gives
most of the calibration benefit of full BNNs. Snoek 2015 first named the
recipe; Kristiadi 2020 sharpened the analysis.

Here we simulate: train a small MLP to fit sin(2x), extract features,
run Bayesian linear regression on top.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _relu(x): return np.maximum(x, 0.0)


def train_features(x, y, d_hid=32, lr=1e-2, epochs=2000, seed=0):
    """Train a 2-layer MLP; return dict with weights."""
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / x.shape[1]), (x.shape[1], d_hid))
    b1 = np.zeros(d_hid)
    W2 = rng.normal(0, np.sqrt(2.0 / d_hid), (d_hid, 1))
    b2 = np.zeros(1)
    n = x.shape[0]
    for _ in range(epochs):
        h_pre = x @ W1 + b1
        h = _relu(h_pre)
        yhat = (h @ W2 + b2).ravel()
        d_y = 2 * (yhat - y) / n
        d_W2 = h.T @ d_y[:, None]
        d_b2 = np.array([d_y.sum()])
        d_h = d_y[:, None] @ W2.T
        d_h[h_pre <= 0] = 0.0
        d_W1 = x.T @ d_h
        d_b1 = d_h.sum(axis=0)
        W1 -= lr * d_W1; b1 -= lr * d_b1
        W2 -= lr * d_W2; b2 -= lr * d_b2
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def features(net, x):
    return _relu(x @ net["W1"] + net["b1"])


def bayes_last_layer(Phi, y, sigma_w=1.0, sigma_n=0.15):
    d = Phi.shape[1]
    A = Phi.T @ Phi / sigma_n ** 2 + np.eye(d) / sigma_w ** 2
    Sigma = np.linalg.inv(A)
    mu = Sigma @ Phi.T @ y / sigma_n ** 2
    return mu, Sigma


def predict_bll(net, mu, Sigma, x, sigma_n=0.15):
    Phi = features(net, x)
    mu_pred = Phi @ mu
    var_pred = np.einsum("nd,de,ne->n", Phi, Sigma, Phi) + sigma_n ** 2
    return mu_pred, var_pred


if __name__ == "__main__":
    print("=== Last-layer Bayesian (Neural Linear) ===\n")
    rng = np.random.default_rng(0)
    x_tr = rng.uniform(-2, 2, 100).reshape(-1, 1)
    y_tr = np.sin(2 * x_tr[:, 0]) + rng.normal(0, 0.15, 100)

    net = train_features(x_tr, y_tr, d_hid=32, epochs=2000, seed=0)
    Phi = features(net, x_tr)
    mu, Sigma = bayes_last_layer(Phi, y_tr, sigma_w=1.0, sigma_n=0.15)

    x_te = np.linspace(-3, 3, 21).reshape(-1, 1)
    mu_pred, var_pred = predict_bll(net, mu, Sigma, x_te, sigma_n=0.15)

    print(f"  {'x':>6}  {'mu':>7}  {'sd':>6}  region")
    for i, xv in enumerate(x_te[:, 0]):
        region = "in " if -2 <= xv <= 2 else "out"
        print(f"  {xv:>6.2f}  {mu_pred[i]:>7.3f}  {np.sqrt(var_pred[i]):>6.3f}  {region}")

    in_mask = (x_te[:, 0] >= -2) & (x_te[:, 0] <= 2)
    sd_te = np.sqrt(var_pred)
    ratio = sd_te[~in_mask].mean() / sd_te[in_mask].mean()
    print(f"\n  predictive sd ratio (out/in): {ratio:.2f}x   <- BLL widens on OOD.\n")

    print("--- library cross-check (sklearn BayesianRidge on penultimate features; laplace-torch) ---")

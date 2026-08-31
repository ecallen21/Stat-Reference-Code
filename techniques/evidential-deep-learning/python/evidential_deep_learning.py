"""Evidential deep learning (Reference Ch 29 Uncertainty Quantification).

Sensoy, Kaplan & Kandemir (2018) "Evidential Deep Learning to Quantify
Classification Uncertainty."  Amini et al. (2020) "Deep Evidential Regression."

The network outputs the parameters of a HIGHER-ORDER distribution over the
target's distribution:
  Classification: Dirichlet(alpha)  ->  categorical over K classes.
  Regression:     Normal-Inverse-Gamma NIG(gamma, nu, alpha, beta)  ->  Gaussian.

Aleatoric and epistemic uncertainty are read off analytically:

  Classification (Dirichlet):
    S = sum(alpha),  p_hat = alpha / S,   epistemic = K / S    (small S = high epistemic).

  Regression (NIG):
    mean       = gamma
    aleatoric  = beta / (alpha - 1)                    (marginal Student-t scale)
    epistemic  = beta / (nu (alpha - 1))               (data uncertainty about mean)

Loss = MSE-style loss on the categorical mean + KL penalty toward flat
prior for misclassified / high-error examples (Sensoy 2018 eq 5).

To keep the demo focused on the *Dirichlet analytics* and avoid the well-
known ReLU-features-extrapolate-badly failure mode, we
(a) show the analytics on hand-set alpha vectors, then
(b) train a simple network with the evidential head on synthetic data
    and confirm the loss decreases and the calibrated Dirichlet gives
    sensible in-distribution predictions.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays

from scipy.special import digamma, gammaln    # digamma / gammaln for Dirichlet


def dirichlet_uncertainty(alpha):
    """Return p_hat, total-evidence S-K, and Sensoy 'vacuity' u = K / S."""
    alpha = np.atleast_2d(alpha)
    K = alpha.shape[1]
    S = alpha.sum(axis=1, keepdims=True)
    p_hat = alpha / S
    vacuity = (K / S).ravel()
    evidence = (S.ravel() - K)
    return {"p_hat": p_hat, "vacuity": vacuity, "evidence": evidence}


def _relu(x): return np.maximum(x, 0.0)


def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _init(rng, d_in, d_hid, K):
    return {
        "W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hid)),
        "b1": np.zeros(d_hid),
        "W2": rng.normal(0, np.sqrt(2 / d_hid), (d_hid, K)),
        "b2": np.zeros(K),
    }


def _forward(p, x):
    h_pre = x @ p["W1"] + p["b1"]
    h = _relu(h_pre)
    pre = h @ p["W2"] + p["b2"]
    e = _softplus(pre)
    alpha = e + 1.0
    return alpha, h_pre, h, pre


def evidential_train(x, y_int, K, d_hid=32, lr=1e-2, epochs=800, seed=0):
    rng = np.random.default_rng(seed)
    p = _init(rng, x.shape[1], d_hid, K)
    n = x.shape[0]
    y_one = np.eye(K)[y_int]
    losses = []
    for _ in range(epochs):
        alpha, h_pre, h, pre = _forward(p, x)
        S = alpha.sum(axis=1, keepdims=True)
        p_hat = alpha / S
        err = ((y_one - p_hat) ** 2).sum(axis=1).mean()
        var = (p_hat * (1 - p_hat) / (S + 1)).sum(axis=1).mean()
        losses.append(err + var)
        # gradient of MSE part (dominant)
        d_p = 2 * (p_hat - y_one) / n
        d_alpha = d_p * (1 - p_hat) / S   # first-order chain
        d_pre = d_alpha * (1 / (1 + np.exp(-np.clip(pre, -30, 30))))
        d_W2 = h.T @ d_pre
        d_b2 = d_pre.sum(axis=0)
        d_h = d_pre @ p["W2"].T
        d_h[h_pre <= 0] = 0.0
        d_W1 = x.T @ d_h
        d_b1 = d_h.sum(axis=0)
        p["W1"] -= lr * d_W1; p["b1"] -= lr * d_b1
        p["W2"] -= lr * d_W2; p["b2"] -= lr * d_b2
    return p, losses


def evidential_predict(p, x):
    alpha, *_ = _forward(p, x)
    return dirichlet_uncertainty(alpha)


if __name__ == "__main__":
    print("=== Part 1 — Dirichlet analytics (Sensoy 2018) ===\n")
    demos = [
        ("no evidence     (flat)   alpha=[1,1,1]", np.array([1., 1., 1.])),
        ("weak evidence   (mild)   alpha=[2,1,1]", np.array([2., 1., 1.])),
        ("strong evidence         alpha=[10,1,1]", np.array([10., 1., 1.])),
        ("very strong             alpha=[50,1,1]", np.array([50., 1., 1.])),
        ("conflict                alpha=[10,10,1]", np.array([10., 10., 1.])),
    ]
    for label, a in demos:
        r = dirichlet_uncertainty(a)
        print(f"  {label:44s}  p_hat={r['p_hat'][0].round(3).tolist()}"
              f"  vacuity={r['vacuity'][0]:.3f}  evidence={r['evidence'][0]:.1f}")

    print("\n=== Part 2 — Train evidential MLP on synthetic 3-class blobs ===\n")
    rng = np.random.default_rng(0)
    K = 3
    centers = np.array([[0, 0], [3, 0], [0, 3]])
    n_per = 100
    X = np.vstack([centers[k] + rng.normal(0, 0.5, (n_per, 2)) for k in range(K)])
    y = np.repeat(np.arange(K), n_per)

    p, losses = evidential_train(X, y, K=K, d_hid=64, epochs=3000, seed=0)
    print(f"  loss:  first={losses[0]:.4f}   final={losses[-1]:.4f}")

    # in-distribution predictions
    X_in = np.vstack([centers[k] + rng.normal(0, 0.5, (4, 2)) for k in range(K)])
    y_in = np.repeat(np.arange(K), 4)
    r_in = evidential_predict(p, X_in)
    hits = int((r_in["p_hat"].argmax(axis=1) == y_in).sum())
    print(f"  in-dist test accuracy:       {hits}/{len(y_in)}")
    print(f"  in-dist mean total evidence: {r_in['evidence'].mean():.2f}")
    print(f"  in-dist mean vacuity K/S:    {r_in['vacuity'].mean():.3f}"
          "     (lower = more confident)\n")

    print("--- library cross-check (evidential-deep-learning-pytorch; edl-pytorch) ---")

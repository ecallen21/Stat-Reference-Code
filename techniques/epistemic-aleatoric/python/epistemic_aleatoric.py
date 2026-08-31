"""Epistemic vs aleatoric uncertainty decomposition (Reference Ch 29 UQ).

Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep
Learning for Computer Vision?"  Depeweg et al. (2018) formalise the
information-theoretic decomposition.

Given an ensemble / posterior sample of models {f_k(x) -> (mu_k, sigma_k^2)},
the predictive distribution is a mixture. Its total variance decomposes:

  Var_total(x) = E_k[sigma_k^2(x)]     +   Var_k[mu_k(x)]
                 <- ALEATORIC (data noise) ->     <- EPISTEMIC (model knowledge) ->

Classification (predictive entropy decomposition, Depeweg 2018):

  H_total(x) = H[E_k p_k(y|x)]         = expected entropy    + mutual information
             = E_k H[p_k(y|x)]   +   I[y ; theta | x, D]
                <- aleatoric ->     <- epistemic ->

The mutual information is the BALD acquisition function (Houlsby 2011)
used in active learning.

Behaviour:
  - Aleatoric: constant with N; SHRINKS with a better model that captures
    heteroscedasticity; irreducible in a single-input regime.
  - Epistemic: DECREASES with more training data; INCREASES away from
    training support.

Here we demonstrate both decompositions on synthetic data:
  Part 1: regression with heteroscedastic noise + a small deep ensemble;
          plot aleatoric vs epistemic across the input range.
  Part 2: 3-class softmax classifier and the expected-entropy /
          mutual-information split on in-distribution vs OOD points.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _relu(x): return np.maximum(x, 0.0)


def _mlp_init(rng, d_in=1, d_hid=32, d_out=2):
    return {"W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hid)),
             "b1": np.zeros(d_hid),
             "W2": rng.normal(0, np.sqrt(2 / d_hid), (d_hid, d_out)),
             "b2": np.zeros(d_out)}


def _mlp_forward(p, x):
    h = _relu(x @ p["W1"] + p["b1"])
    out = h @ p["W2"] + p["b2"]
    mu = out[:, 0]
    lv = np.clip(out[:, 1], -6, 6)
    return mu, np.exp(lv), h


def train_member(x, y, rng, epochs=1500, lr=1e-2, d_hid=32):
    p = _mlp_init(rng, x.shape[1], d_hid, 2)
    n = x.shape[0]
    for _ in range(epochs):
        mu, var, h = _mlp_forward(p, x)
        d_mu = -(y - mu) / var / n
        d_lv = 0.5 * (1 - (y - mu) ** 2 / var) / n
        d_out = np.stack([d_mu, d_lv], axis=1)
        d_W2 = h.T @ d_out
        d_b2 = d_out.sum(axis=0)
        d_h = d_out @ p["W2"].T
        d_h[h <= 0] = 0.0
        d_W1 = x.T @ d_h
        d_b1 = d_h.sum(axis=0)
        p["W1"] -= lr * d_W1; p["b1"] -= lr * d_b1
        p["W2"] -= lr * d_W2; p["b2"] -= lr * d_b2
    return p


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropy(p, eps=1e-30):
    return -np.sum(p * np.log(p + eps), axis=-1)


if __name__ == "__main__":
    print("=== Part 1 — Regression: aleatoric + epistemic decomposition ===\n")
    rng = np.random.default_rng(0)
    x_tr = rng.uniform(-2, 2, 100).reshape(-1, 1)
    noise = 0.1 + 0.4 * np.abs(x_tr[:, 0])      # heteroscedastic noise
    y_tr = np.sin(1.5 * x_tr[:, 0]) + rng.normal(0, noise)

    members = [train_member(x_tr, y_tr,
                             np.random.default_rng(k), d_hid=32, epochs=1500)
                for k in range(5)]

    x_te = np.linspace(-3, 3, 15).reshape(-1, 1)
    mus, vars_ = [], []
    for p in members:
        m, v, _ = _mlp_forward(p, x_te)
        mus.append(m); vars_.append(v)
    mus = np.array(mus); vars_ = np.array(vars_)
    aleatoric = vars_.mean(axis=0)
    epistemic = mus.var(axis=0)
    total = aleatoric + epistemic

    print(f"  {'x':>6}  {'total_sd':>8}  {'aleat_sd':>8}  {'epist_sd':>8}  region")
    for i, xv in enumerate(x_te[:, 0]):
        region = "in " if -2 <= xv <= 2 else "out"
        print(f"  {xv:>6.2f}  {np.sqrt(total[i]):>8.3f}  {np.sqrt(aleatoric[i]):>8.3f}"
              f"  {np.sqrt(epistemic[i]):>8.3f}  {region}")

    in_mask = (x_te[:, 0] >= -2) & (x_te[:, 0] <= 2)
    print(f"\n  in-dist:  aleat={np.sqrt(aleatoric[in_mask]).mean():.3f}"
          f"  epist={np.sqrt(epistemic[in_mask]).mean():.3f}")
    print(f"  OOD    :  aleat={np.sqrt(aleatoric[~in_mask]).mean():.3f}"
          f"  epist={np.sqrt(epistemic[~in_mask]).mean():.3f}"
          "     <- epistemic should climb OOD; aleatoric is a data property.")

    print("\n=== Part 2 — Classification: predictive entropy = expected + mutual info ===\n")
    rng = np.random.default_rng(1)
    K = 3
    # Ensemble of K_ens classifiers, each outputs a softmax over K classes.
    # We synthesise two cases: (A) ambiguous / high aleatoric; (B) OOD / high epistemic.
    def ensemble_probs(case):
        if case == "in-dist confident":
            # All models agree on a sharp answer -> low both.
            return np.array([[0.9, 0.05, 0.05]] * 5)
        if case == "in-dist ambiguous":
            # All models agree that y is uncertain (label noise / hard example).
            return np.array([[0.5, 0.3, 0.2]] * 5)
        if case == "OOD disagreement":
            # Each model confidently picks a different class -> epistemic ~ log K.
            return np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9],
                              [0.7, 0.15, 0.15], [0.1, 0.1, 0.8]])
        raise ValueError(case)

    print(f"  {'case':30s}  {'H_total':>8}  {'H_expected(aleat)':>18}  {'MI(epist)':>10}")
    for case in ("in-dist confident", "in-dist ambiguous", "OOD disagreement"):
        pk = ensemble_probs(case)                 # (n_ens, K)
        mean_p = pk.mean(axis=0)                  # marginal predictive
        H_total = entropy(mean_p[None]).item()
        H_expected = entropy(pk).mean()           # E_k H[p_k]  (aleatoric)
        MI = H_total - H_expected                 # I[y ; theta | x] (epistemic)
        print(f"  {case:30s}  {H_total:>8.3f}  {H_expected:>18.3f}  {MI:>10.3f}")

    print("\n  MI > 0 whenever the ensemble members disagree -- epistemic uncertainty.\n")
    print("--- library cross-check (uncertainty-toolbox; laplace-torch; pyro) ---")

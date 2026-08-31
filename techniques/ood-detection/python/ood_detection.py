"""Out-of-distribution (OOD) detection (Reference Ch 29 UQ).

Score every test input by how INLIER it is; threshold to flag OOD.
Baselines from the literature:

  1. MSP  -- Maximum Softmax Probability (Hendrycks-Gimpel 2017)
     score(x) = max_k p_hat_k(x).   OOD if score < threshold.

  2. Energy (Liu-Zhang-Owens-Li 2020)
     E(x) = -T * log sum_k exp(z_k(x) / T)
     OOD if E(x) > threshold (higher energy = less confident).

  3. Mahalanobis distance (Lee-Lee-Lee-Shin 2018)
     Fit class-conditional Gaussian on penultimate features:
       mu_k = mean_{i:y_i=k} phi(x_i);  Sigma = pooled cov
     score(x) = -min_k (phi(x) - mu_k)^T Sigma^-1 (phi(x) - mu_k)
     OOD if score < threshold.

Evaluation:
  - AUROC across in-vs-out labels (higher = better).
  - FPR at 95% TPR: what fraction of OOD points are wrongly kept when we
    accept 95% of the in-distribution inputs.

Here we implement all three baselines and compare AUROC on a synthetic
task with well-separated in-distribution Gaussian clusters vs OOD noise.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def msp_score(logits):
    return _softmax(logits).max(axis=1)      # high = in-distribution


def energy_score(logits, T=1.0):
    z = logits / T
    m = z.max(axis=1, keepdims=True)
    lse = m.ravel() + np.log(np.exp(z - m).sum(axis=1))
    return T * lse                             # high = in-distribution (Liu 2020 uses -E; we align signs)


def mahalanobis_fit(features, y, K):
    mus = np.stack([features[y == k].mean(axis=0) for k in range(K)])
    resid = np.vstack([features[y == k] - mus[k] for k in range(K)])
    cov = (resid.T @ resid) / max(len(resid) - K, 1)
    prec = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
    return mus, prec


def mahalanobis_score(features_new, mus, prec):
    # Score = -min_k Mahalanobis distance^2  (higher = more inlier)
    n = features_new.shape[0]
    K = mus.shape[0]
    dists = np.zeros((n, K))
    for k in range(K):
        d = features_new - mus[k]
        dists[:, k] = np.einsum("nd,de,ne->n", d, prec, d)
    return -dists.min(axis=1)


def auroc(scores, labels):
    """Positive class = in-distribution (label 1)."""
    order = np.argsort(-scores)                  # descending: highest score first
    lab = labels[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    tp = np.concatenate(([0], tp))
    fp = np.concatenate(([0], fp))
    tp = tp / tp[-1] if tp[-1] > 0 else tp
    fp = fp / fp[-1] if fp[-1] > 0 else fp
    return float(np.trapezoid(tp, fp))


def fpr_at_tpr(scores, labels, tpr_target=0.95):
    """FPR (fraction of OOD kept as in-dist) when accepting tpr_target of in-dist."""
    in_scores = scores[labels == 1]
    out_scores = scores[labels == 0]
    thr = np.quantile(in_scores, 1 - tpr_target)  # accept if score >= thr
    return float((out_scores >= thr).mean())


if __name__ == "__main__":
    print("=== OOD detection: MSP, Energy, Mahalanobis ===\n")
    rng = np.random.default_rng(0)
    K = 4
    d = 8
    centers = rng.normal(0, 3, (K, d))
    n_per, n_test_in, n_test_out = 100, 200, 200

    # in-distribution training features + class labels
    y_tr = np.repeat(np.arange(K), n_per)
    Phi_tr = np.vstack([centers[k] + rng.normal(0, 0.7, (n_per, d)) for k in range(K)])
    # "logits" = signed distance to each centre (proxy for a trained head)
    def logits(features):
        return -np.array([((features - centers[k]) ** 2).sum(axis=1) for k in range(K)]).T

    # in-distribution test
    y_te_in = np.repeat(np.arange(K), n_test_in // K)
    Phi_te_in = np.vstack([centers[k] + rng.normal(0, 0.7, (n_test_in // K, d)) for k in range(K)])
    # OOD test: uniform far-field noise
    Phi_te_out = rng.uniform(-15, 15, (n_test_out, d))

    Phi_all = np.vstack([Phi_te_in, Phi_te_out])
    labels = np.concatenate([np.ones(len(Phi_te_in)), np.zeros(len(Phi_te_out))])

    msp = msp_score(logits(Phi_all))
    en  = energy_score(logits(Phi_all), T=1.0)
    mus, prec = mahalanobis_fit(Phi_tr, y_tr, K)
    ma = mahalanobis_score(Phi_all, mus, prec)

    print(f"  {'method':16s}  {'AUROC':>7}  {'FPR@95%TPR':>11}")
    for name, s in (("MSP",  msp), ("Energy", en), ("Mahalanobis", ma)):
        print(f"  {name:16s}  {auroc(s, labels):>7.3f}  {fpr_at_tpr(s, labels):>11.3f}")

    print("\n  Mahalanobis usually wins when penultimate features are Gaussian-like.\n")
    print("--- library cross-check (pytorch-ood, cleanlab OOD, torchdrift) ---")

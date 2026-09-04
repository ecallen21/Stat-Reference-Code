"""Structural topic model (Reference Sec 42.7).

Roberts-Stewart-Tingley 2019.  Extension of LDA that lets DOCUMENT-
LEVEL COVARIATES influence:
  * Topic PREVALENCE  (probability of using a topic; document-topic).
  * Topic CONTENT     (word choice within a topic; topic-word).

Compact demo: 2 known topics + 2 groups with shifted topic prevalence.
Fit a stripped LDA-like model (EM on multinomial mixture) with a
per-group topic-prevalence prior; report topic-word distributions
and group-level topic prevalences.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def _e_step(X, phi, theta):
    """E-step: posterior over topics for each document-word."""
    D, V = X.shape; K = phi.shape[0]
    posts = []
    for d in range(D):
        # For each word type, P(k | word) proportional to theta[d, k] * phi[k, w]
        p = theta[d][:, None] * phi        # (K, V)
        p = p / (p.sum(axis=0, keepdims=True) + 1e-12)
        posts.append(p)
    return posts


def stm_em(X, group, K=2, alpha_group=None, n_iter=60, seed=0):
    """Toy STM: EM with group-specific alpha prior on document-topic distribution."""
    rng = np.random.default_rng(seed)
    D, V = X.shape
    groups = np.unique(group)
    G = len(groups)
    # Initialise
    phi = rng.dirichlet(np.ones(V), size=K)
    theta = rng.dirichlet(np.ones(K), size=D)
    alpha = alpha_group if alpha_group is not None else np.ones((G, K))
    for _ in range(n_iter):
        posts = _e_step(X, phi, theta)
        # M-step: update phi and theta
        new_phi = np.zeros_like(phi)
        for d, post in enumerate(posts):
            new_phi += post * X[d]
        new_phi = new_phi / new_phi.sum(axis=1, keepdims=True)
        new_theta = np.zeros_like(theta)
        for d, post in enumerate(posts):
            n_kd = (post * X[d]).sum(axis=1)
            g = list(groups).index(group[d])
            new_theta[d] = (n_kd + alpha[g] - 1) / (n_kd.sum() + alpha[g].sum() - K)
            new_theta[d] = np.clip(new_theta[d], 1e-6, 1.0)
            new_theta[d] /= new_theta[d].sum()
        phi = new_phi; theta = new_theta
    # Estimate per-group topic prevalence
    prev = np.stack([theta[group == g].mean(axis=0) for g in groups], axis=0)
    return {"phi": phi, "theta": theta, "prevalence_by_group": prev, "groups": list(groups)}


if __name__ == "__main__":
    print("=== Structural topic model: covariate-aware topic prevalence ===\n")
    rng = np.random.default_rng(0)
    vocab = ["health", "doctor", "hospital", "sport", "team", "game"]
    V = len(vocab)
    # Two true topics: medical and sport
    true_phi = np.array([[0.35, 0.30, 0.25, 0.03, 0.04, 0.03],
                          [0.02, 0.03, 0.05, 0.30, 0.30, 0.30]])
    D = 60
    group = np.repeat([0, 1], D // 2)     # covariate: publication venue
    # Group 0 tends to be medical (topic 0), group 1 sport (topic 1)
    theta_true = np.zeros((D, 2))
    for d in range(D):
        if group[d] == 0:
            theta_true[d] = rng.dirichlet([5, 1])
        else:
            theta_true[d] = rng.dirichlet([1, 5])
    X = np.zeros((D, V))
    for d in range(D):
        w_dist = theta_true[d] @ true_phi
        counts = rng.multinomial(30, w_dist)
        X[d] = counts

    # Fit STM with per-group prevalence prior
    alpha = np.array([[3.0, 1.0], [1.0, 3.0]])
    res = stm_em(X, group, K=2, alpha_group=alpha)

    print("  Topic-word distributions (top words per topic):")
    for k in range(2):
        top = np.argsort(-res["phi"][k])[:4]
        print(f"    topic {k}: " + "  ".join(f"{vocab[i]}={res['phi'][k, i]:.2f}" for i in top))

    print(f"\n  Per-group topic prevalence:")
    for g, row in zip(res["groups"], res["prevalence_by_group"]):
        print(f"    group {g}:  " + "  ".join(f"topic{k}={p:.2f}" for k, p in enumerate(row)))

    print("\n--- library cross-check (R stm::stm/estimateEffect; Python custom + bertopic) ---")

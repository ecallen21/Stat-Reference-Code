"""Wordfish scaling (Reference Sec 42.18).

Slapin & Proksch (2008).  UNSUPERVISED positioning of documents on a
single latent dimension using word counts:

    y_{ij} ~ Poisson(exp(alpha_i + psi_j + beta_j * omega_i))

  alpha_i : document i verbosity offset
  omega_i : document i position on the latent scale
  psi_j   : word j baseline frequency
  beta_j  : word j discrimination on the latent scale

Estimated by alternating Poisson regressions (documents fixed while
updating words, and vice versa) with an identification constraint
(e.g., omega centred, unit variance).

Widely used in political-text scaling; independent of pre-specified
"reference texts" that Wordscores requires.
"""
from __future__ import annotations    # stdlib

import re

import numpy as np    # numerical arrays


def _dtm(docs):
    tok = [re.findall(r"\w+", d.lower()) for d in docs]
    vocab = sorted({w for d in tok for w in d})
    idx = {w: i for i, w in enumerate(vocab)}
    Y = np.zeros((len(docs), len(vocab)), dtype=int)
    for i, t in enumerate(tok):
        for w in t: Y[i, idx[w]] += 1
    return Y, vocab


def wordfish(Y, n_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    n, V = Y.shape
    omega = rng.normal(0, 1, n)
    alpha = np.log(Y.sum(axis=1).clip(1))
    psi = np.log(Y.sum(axis=0).clip(1) / n)
    beta = rng.normal(0, 0.1, V)
    for _ in range(n_iter):
        # Update word (psi, beta) via Poisson likelihood M-step: gradient step
        eta = alpha[:, None] + psi[None, :] + omega[:, None] * beta[None, :]
        mu = np.exp(eta)
        grad_psi = (Y - mu).sum(axis=0)
        hess_psi = -mu.sum(axis=0)
        psi = psi - grad_psi / hess_psi
        eta = alpha[:, None] + psi[None, :] + omega[:, None] * beta[None, :]
        mu = np.exp(eta)
        grad_beta = ((Y - mu) * omega[:, None]).sum(axis=0)
        hess_beta = -(mu * omega[:, None] ** 2).sum(axis=0)
        beta = beta - grad_beta / hess_beta
        # Update document (alpha, omega)
        eta = alpha[:, None] + psi[None, :] + omega[:, None] * beta[None, :]
        mu = np.exp(eta)
        grad_alpha = (Y - mu).sum(axis=1)
        hess_alpha = -mu.sum(axis=1)
        alpha = alpha - grad_alpha / hess_alpha
        eta = alpha[:, None] + psi[None, :] + omega[:, None] * beta[None, :]
        mu = np.exp(eta)
        grad_omega = ((Y - mu) * beta[None, :]).sum(axis=1)
        hess_omega = -(mu * beta[None, :] ** 2).sum(axis=1)
        omega = omega - grad_omega / hess_omega
        # Identification: omega mean 0, sd 1
        omega = (omega - omega.mean()) / max(omega.std(), 1e-6)
    return {"omega": omega, "alpha": alpha, "psi": psi, "beta": beta}


if __name__ == "__main__":
    print("=== Wordfish: unsupervised Poisson scaling of documents ===\n")
    # Left-leaning and right-leaning "party" documents
    docs = [
        "worker union labor solidarity strike",
        "worker rights labor equality reform",
        "market freedom liberty enterprise growth",
        "market taxes economy investment jobs",
        "worker labor market growth reform",   # centrist
    ]
    Y, vocab = _dtm(docs)
    r = wordfish(Y)
    print(f"  Estimated document positions (omega):")
    for i, o in enumerate(r["omega"]):
        print(f"    doc {i}: omega = {o:+.3f}   ({docs[i]!r})")

    print(f"\n  Top 3 words with largest positive beta (rightward end):")
    top = np.argsort(-r["beta"])[:3]
    for j in top:
        print(f"    {vocab[j]:<12s}  beta = {r['beta'][j]:+.3f}")
    print(f"  Top 3 words with largest negative beta (leftward end):")
    bot = np.argsort(r["beta"])[:3]
    for j in bot:
        print(f"    {vocab[j]:<12s}  beta = {r['beta'][j]:+.3f}")

    print("\n--- library cross-check (R quanteda.textmodels::textmodel_wordfish; Python custom + textacy) ---")

"""Bock nominal response model (Reference §22.x extra).

For an item with K unordered response categories:

    P(U_ij = k | theta_i) = exp( a_{jk} theta_i + c_{jk} )
                            / sum_l exp( a_{jl} theta_i + c_{jl} )

Identifiability constraints: a_{j0} = 0 and c_{j0} = 0 for the reference
category.  The K - 1 remaining (a, c) pairs are free per item.

Used for multiple-choice items where distractors carry no ordinal
information — the response IS the diagnostic, not a monotone score.

We fit by MML with Gauss-Hermite quadrature (Bock-Aitkin style).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt, pi)

import numpy as np    # numerical arrays + linear algebra


def _softmax_rows(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)


def _fit_item_multinomial(theta, u_int, K, n_iter: int = 30):
    """Fit (a, c) for one item: unordered multinomial logit vs theta."""
    n = len(theta)
    # design: [theta, 1] per non-reference category; params (K-1) x 2
    A = np.zeros(K - 1); C = np.zeros(K - 1)
    Y = np.zeros((n, K))
    Y[np.arange(n), u_int] = 1
    for _ in range(n_iter):
        Z = np.zeros((n, K))
        Z[:, 1:] = np.outer(theta, A) + C[None, :]
        P = _softmax_rows(Z)
        # gradient for A[k] = sum_i theta_i (Y[i,k+1] - P[i,k+1])
        gA = theta @ (Y[:, 1:] - P[:, 1:])
        gC = (Y[:, 1:] - P[:, 1:]).sum(axis=0)
        # diagonal Hessian approximation (block-diagonal per category)
        w = P[:, 1:] * (1 - P[:, 1:]) + 1e-6
        HA = (theta ** 2)[:, None] * w                       # (n, K-1)
        HAd = HA.sum(axis=0); HCd = w.sum(axis=0)
        HAC = (theta[:, None] * w).sum(axis=0)
        # solve per category
        for k in range(K - 1):
            H = np.array([[HAd[k], HAC[k]], [HAC[k], HCd[k]]])
            step = np.linalg.solve(H + 1e-3 * np.eye(2), np.array([gA[k], gC[k]]))
            step = np.clip(step, -0.5, 0.5)
            A[k] += step[0]; C[k] += step[1]
    return A, C


def fit_nominal(U, K: int, n_iter: int = 4, n_quad: int = 21,
                seed: int = 0) -> dict:
    """MML for the nominal response model with N(0,1) latent trait."""
    U = np.asarray(U, dtype=int); n, J = U.shape
    # warm-start theta from row score / K (approximate)
    theta = (U.mean(axis=1) - U.mean()) / (U.std() + 1e-9)

    a = np.zeros((J, K - 1)); c = np.zeros((J, K - 1))

    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    xs = math.sqrt(2) * nodes; ws = weights / math.sqrt(math.pi)

    for _ in range(n_iter):
        # M-step: per-item multinomial-logit given current EAP thetas
        for j in range(J):
            a[j], c[j] = _fit_item_multinomial(theta, U[:, j], K)
        # E-step: EAP of theta given items
        # log-lik on the quadrature grid: (n_quad, J, K)
        Z = np.zeros((n_quad, J, K))
        Z[:, :, 1:] = xs[:, None, None] * a[None, :, :] + c[None, :, :]
        logP = Z - np.log(np.exp(Z - Z.max(axis=-1, keepdims=True))
                          .sum(axis=-1, keepdims=True)) - Z.max(axis=-1, keepdims=True)
        # For each person i: pick out logP[k, j, U[i, j]] and sum over j
        ll_ik = np.zeros((n, n_quad))
        for i in range(n):
            ll_ik[i] = logP[:, np.arange(J), U[i]].sum(axis=1) + np.log(ws + 1e-12)
        ll_ik -= ll_ik.max(axis=1, keepdims=True)
        post = np.exp(ll_ik); post /= post.sum(axis=1, keepdims=True)
        theta = post @ xs
        # identify: theta ~ mean 0, sd 1; compensate a and c
        mu = float(theta.mean()); theta -= mu; c = c + a * mu
        s = float(theta.std(ddof=1))
        if s > 1e-3:
            theta /= s; a = a * s

    # log-lik
    logL = 0.0
    for i in range(n):
        z = np.zeros((J, K)); z[:, 1:] = theta[i] * a + c
        pi = _softmax_rows(z)
        logL += float(np.log(pi[np.arange(J), U[i]] + 1e-12).sum())
    return {"theta": theta, "a": a, "c": c, "K": K, "log_lik": logL,
            "method": "nominal response model (Bock 1972) via MML + Gauss-Hermite"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500; J = 8; K = 4
    theta_true = rng.normal(size=n)
    a_true = rng.normal(scale=0.8, size=(J, K - 1))
    c_true = rng.normal(scale=0.5, size=(J, K - 1))

    # simulate
    U = np.zeros((n, J), dtype=int)
    for j in range(J):
        Z = np.zeros((n, K))
        Z[:, 1:] = np.outer(theta_true, a_true[j]) + c_true[j]
        P = _softmax_rows(Z)
        for i in range(n):
            U[i, j] = int(rng.choice(K, p=P[i]))

    fit = fit_nominal(U, K=K, n_iter=4, n_quad=21)
    print(f"=== Bock nominal response model (n={n}, J={J}, K={K}) ===")
    print(f"  log-lik = {fit['log_lik']:.1f}")

    # correlations for a and c across all (item, category) cells
    ra = float(np.corrcoef(fit["a"].ravel(), a_true.ravel())[0, 1])
    rc = float(np.corrcoef(fit["c"].ravel(), c_true.ravel())[0, 1])
    rt = float(np.corrcoef(fit["theta"], theta_true)[0, 1])
    print(f"  cor(a_hat, a_true)      = {ra:+.3f}   ({J*(K-1)} cells)")
    print(f"  cor(c_hat, c_true)      = {rc:+.3f}")
    print(f"  cor(theta_hat, theta)   = {rt:+.3f}")

    print(f"\n  example item j=0 category slopes (a_{{j, k}}):")
    for k in range(K - 1):
        print(f"    k={k+1}: a_hat = {fit['a'][0, k]:+.3f}   "
              f"a_true = {a_true[0, k]:+.3f}")

    print("\n--- library cross-check (R mirt::mirt(itemtype='nominal')) ---")

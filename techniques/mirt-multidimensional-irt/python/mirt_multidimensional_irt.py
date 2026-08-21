"""Multidimensional Item Response Theory: compensatory M2PL (Reference §22.x extra).

Item j has:
  - discrimination vector a_j in R^d       (loading on each latent trait)
  - difficulty scalar b_j
Person i has ability theta_i in R^d.

Compensatory M2PL:
    P(U_ij = 1 | theta_i) = sigma( a_j^T theta_i - b_j )

Non-compensatory (Sympson):
    P(U_ij = 1) = prod_k sigma( a_jk * theta_ik - b_jk )

Fit (this module): PCA warm-start on the tetrachoric-approximate correlation
matrix; per-item logistic regression on the warm-start abilities; final EAP
theta update via 2D Gauss-Hermite quadrature.  Production packages (`mirt`,
`ltm`, `pyIRT`) fit MML by Bock-Aitkin EM with quadrature, or full Bayes;
JML is inconsistent and best avoided.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt, pi, log)

import numpy as np    # numerical arrays + linear algebra


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_item_logistic(theta, u, n_iter: int = 20) -> tuple:
    """Fit (a, b) for one item by IRLS with damped Newton."""
    d = theta.shape[1]
    beta = np.zeros(d + 1)                                # [a; b]
    X = np.column_stack([theta, -np.ones(len(u))])
    for _ in range(n_iter):
        eta = X @ beta; p = _sigmoid(eta)
        g = X.T @ (u - p)
        W = p * (1 - p) + 1e-6
        H = (X.T * W) @ X
        step = np.linalg.solve(H + 1e-4 * np.eye(d + 1), g)
        beta += np.clip(step, -1.0, 1.0)
    a = np.clip(beta[:d], 0.05, 4.0)
    b = float(np.clip(beta[d], -4.0, 4.0))
    return a, b


def fit_m2pl(U, d: int = 2, n_iter: int = 3, n_quad: int = 11,
             seed: int = 0) -> dict:
    U = np.asarray(U, dtype=float); n, J = U.shape

    # PCA warm start for theta (on the centred item response matrix)
    Uc = U - U.mean(axis=0)
    _, S, VT = np.linalg.svd(Uc, full_matrices=False)
    theta = Uc @ VT.T[:, :d]
    theta = (theta - theta.mean(axis=0)) / (theta.std(axis=0, ddof=1) + 1e-9)

    a = np.zeros((J, d)); b = np.zeros(J)
    for _ in range(n_iter):
        # per-item logistic on current theta
        for j in range(J):
            a[j], b[j] = _fit_item_logistic(theta, U[:, j])

        # EAP theta update via 2-D Gauss-Hermite quadrature
        if d == 2:
            nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
            # transform to standard normal: x_i = sqrt(2)*node_i, w_i = weight_i/sqrt(pi)
            xs = math.sqrt(2) * nodes; ws = weights / math.sqrt(math.pi)
            grid = np.array([(x1, x2) for x1 in xs for x2 in xs])
            gw = np.array([w1 * w2 for w1 in ws for w2 in ws])
            # log-lik of each (person, node): (n, K)
            Z = grid @ a.T - b                                 # (K, J)
            logP = -np.logaddexp(0, -Z)                        # log sigmoid
            log1P = -np.logaddexp(0, Z)                        # log(1-sigmoid)
            # for each i: log w_k + sum_j [U[i,j] log P[k,j] + (1-U[i,j]) log 1P[k,j]]
            ll_ik = (U @ logP.T) + ((1 - U) @ log1P.T) + np.log(gw + 1e-12)
            ll_ik -= ll_ik.max(axis=1, keepdims=True)
            post = np.exp(ll_ik); post /= post.sum(axis=1, keepdims=True)
            theta = post @ grid

    # final log-lik under fitted params
    Z = theta @ a.T - b
    P = _sigmoid(Z)
    ll = float((U * np.log(P + 1e-12) + (1 - U) * np.log(1 - P + 1e-12)).sum())
    return {"theta": theta, "a": a, "b": b, "log_lik": ll,
            "method": "compensatory M2PL (PCA warm-start + item-logistic + EAP)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 800; J = 20; d = 2
    theta_true = rng.normal(size=(n, d))
    # planted simple structure: items 0..9 load on dim 0, items 10..19 on dim 1
    a_true = np.zeros((J, d))
    a_true[:10, 0] = np.abs(rng.normal(1.2, 0.3, 10))
    a_true[10:, 1] = np.abs(rng.normal(1.2, 0.3, 10))
    b_true = rng.uniform(-1, 1, J)

    P = _sigmoid(theta_true @ a_true.T - b_true)
    U = (rng.uniform(size=(n, J)) < P).astype(int)

    fit = fit_m2pl(U, d=2, n_iter=3, n_quad=11)
    print(f"=== M2PL fit (n={n}, J={J}, d={d}) ===")
    print(f"  log-lik = {fit['log_lik']:.1f}   "
          f"(true params LL = {float((U*np.log(P+1e-12)+(1-U)*np.log(1-P+1e-12)).sum()):.1f})")

    # rotate estimated loadings to align with true (Procrustes over d x d)
    M = a_true.T @ fit["a"]
    U_, _, Vt = np.linalg.svd(M)
    Q = U_ @ Vt                                              # optimal orthogonal rotation
    a_rot = fit["a"] @ Q.T
    theta_rot = fit["theta"] @ Q.T

    print("\n  cor(a_hat, a_true) after Procrustes rotation:")
    for k in range(d):
        r = float(np.corrcoef(a_rot[:, k], a_true[:, k])[0, 1])
        print(f"    loading dim {k}: r = {r:+.3f}")
    print("  cor(theta_hat, theta_true) after Procrustes rotation:")
    for k in range(d):
        r = float(np.corrcoef(theta_rot[:, k], theta_true[:, k])[0, 1])
        print(f"    ability dim {k}: r = {r:+.3f}")

    print("\n--- library cross-check (R mirt::mirt(model='F1 = 1-10; F2 = 11-20')) ---")

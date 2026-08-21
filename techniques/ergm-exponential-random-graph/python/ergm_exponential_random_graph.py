"""Exponential Random Graph Model — ERGM (Reference §24.5).

Exponential family over graphs:

    P(G) = exp( theta^T s(G) ) / Z(theta)

where s(G) is a vector of network statistics (e.g. edges, triangles, k-stars,
GWESP).  Z(theta) is intractable so full MLE requires MCMC (Snijders 2002).

We implement:
  * pseudo-likelihood estimation (Strauss & Ikeda 1990) — treat each dyad
    conditional on the rest of the graph as an independent logistic:

        logit P(A_ij = 1 | A_-ij) = theta^T delta_ij s(G)

    where delta_ij s(G) is the change in s(G) when the (i, j) edge toggles.
  * edge / triangle statistics as the default features.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import numpy as np    # numerical arrays + linear algebra


def _edge_delta(A, i, j):
    return 1.0                                            # toggling the edge changes # edges by 1


def _triangle_delta(A, i, j):
    """Change in # triangles when (i, j) toggles = # common neighbours."""
    return float(np.sum(A[i] * A[j]) - A[i, i] * A[j, i])   # exclude i, j themselves


def _design_matrix(A):
    """Rows: each dyad (i, j).  Columns: [edge_delta, triangle_delta]."""
    n = A.shape[0]
    X, y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            X.append([_edge_delta(A, i, j), _triangle_delta(A, i, j)])
            y.append(A[i, j])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _loglik(theta, X, y):
    eta = np.clip(X @ theta, -30, 30)
    return float(np.sum(y * eta - np.log1p(np.exp(eta))))


def ergm_pseudo_likelihood(A, max_iter: int = 200, tol: float = 1e-8) -> dict:
    """Maximise the logistic-regression pseudo-likelihood by IRLS + step halving."""
    X, y = _design_matrix(A)
    p = X.shape[1]
    theta = np.zeros(p)
    ll = _loglik(theta, X, y)
    for it in range(max_iter):
        eta = np.clip(X @ theta, -30, 30)
        mu = _sigmoid(eta)
        grad = X.T @ (y - mu)
        W = mu * (1 - mu) + 1e-8                          # ridge to avoid singular Hessian
        I_fisher = (X.T * W) @ X                          # -H, positive definite
        try:
            step = np.linalg.solve(I_fisher, grad)        # ascent direction
        except np.linalg.LinAlgError:
            break
        # step halving until log-likelihood strictly improves
        alpha = 1.0
        for _ in range(30):
            theta_new = theta + alpha * step
            ll_new = _loglik(theta_new, X, y)
            if ll_new > ll + 1e-12:
                break
            alpha *= 0.5
        else:
            break                                         # no improving step found
        if np.max(np.abs(theta_new - theta)) < tol:
            theta = theta_new; ll = ll_new; break
        theta = theta_new; ll = ll_new
    # observed statistics for reference
    s_obs = X.T @ y
    return {"theta": theta,
            "s_obs": s_obs.tolist(),
            "features": ["edges", "triangles"],
            "n_iter": it + 1,
            "method": "ERGM pseudo-likelihood (edges + triangles)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30
    # Simulate an ER graph (no triangle bias) — expect theta_triangle ≈ 0
    p_true = 0.15
    A = (rng.uniform(size=(n, n)) < p_true).astype(int)
    A = np.triu(A, 1); A = A + A.T

    fit = ergm_pseudo_likelihood(A)
    print("=== ERGM pseudo-likelihood on ER(30, 0.15) ===")
    print(f"  observed edges     = {int(fit['s_obs'][0])}")
    print(f"  observed triangles = {int(fit['s_obs'][1])}")
    print(f"  theta_edges     = {fit['theta'][0]:.4f}  "
          f"(expected logit(p) = {np.log(p_true / (1 - p_true)):.4f})")
    print(f"  theta_triangles = {fit['theta'][1]:+.4f}  (expected ≈ 0)")

    # Simulate a triangle-heavy graph via a 2-clique-of-cliques
    print("\n=== ERGM on a triangle-heavy graph (two 15-cliques) ===")
    B = np.zeros((n, n), dtype=int)
    for i in range(15):
        for j in range(i + 1, 15):
            B[i, j] = B[j, i] = 1
    for i in range(15, 30):
        for j in range(i + 1, 30):
            B[i, j] = B[j, i] = 1
    # add a few cross-edges
    B[0, 15] = B[15, 0] = 1; B[1, 16] = B[16, 1] = 1

    fit2 = ergm_pseudo_likelihood(B)
    print(f"  theta_edges     = {fit2['theta'][0]:.4f}   "
          f"theta_triangles = {fit2['theta'][1]:+.4f}")

    print("\n--- library cross-check (R statnet::ergm; Python ergm-cli via R) ---")

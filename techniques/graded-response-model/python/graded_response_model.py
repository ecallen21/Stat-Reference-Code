"""Samejima's Graded Response Model (GRM) for ordered polytomous items
(Reference §22.7; Samejima 1969).

Extension of 2PL to ITEMS WITH ORDERED CATEGORIES (e.g. Likert 1-5).

For an item with K categories (0, 1, ..., K-1), define K-1 CUMULATIVE
probabilities of scoring at least k:
    P_j^*(k | theta) = 1 / (1 + exp(-a_j (theta - b_jk))),  k = 1, ..., K-1
    P_j^*(0 | theta) = 1
    P_j^*(K | theta) = 0

Category probability:
    P_j(k | theta) = P_j^*(k | theta) - P_j^*(k+1 | theta)

Constraint: b_j1 < b_j2 < ... < b_j,K-1 (thresholds ordered) - implicitly
holds in the GRM since P^* is monotone in b.

Marginal MLE via Gauss-Hermite quadrature.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _sigmoid(x): return 1 / (1 + np.exp(-x))


def grm_mml(Y, K: int, n_quad: int = 15) -> dict:
    """Samejima GRM MML for a single-scale item set with K categories 0..K-1."""
    Y = np.asarray(Y, dtype=int); n, J = Y.shape
    q_nodes, q_weights = np.polynomial.hermite_e.hermegauss(n_quad)
    q_weights = q_weights / math.sqrt(2 * math.pi)
    # Parameters per item: log(a) + K-1 thresholds (ordered via cumulative shifts)
    Ktm = K - 1
    def unpack(params):
        a = np.exp(params[:J])
        b = np.zeros((J, Ktm))
        offset = J
        for j in range(J):
            # Enforce ordering via cumulative-softplus of raw values
            raw = params[offset:offset + Ktm]
            b[j, 0] = raw[0]
            for m in range(1, Ktm): b[j, m] = b[j, m - 1] + math.log(1 + math.exp(raw[m]))
            offset += Ktm
        return a, b

    def cat_prob(a_j, b_j, theta):
        """Return (K x n_theta) matrix of category probabilities."""
        # Cumulative P*(k | theta) for k = 1..Ktm
        P_cum = np.stack([_sigmoid(a_j * (theta - b_j[m])) for m in range(Ktm)])  # (Ktm, n_theta)
        P_cum = np.vstack([np.ones_like(theta)[None, :], P_cum, np.zeros_like(theta)[None, :]])  # (K+1, n_theta)
        return P_cum[:-1] - P_cum[1:]  # (K, n_theta)

    def neg_ll(params):
        a, b = unpack(params)
        ll = 0.0
        for i in range(n):
            probs_all = np.ones(n_quad)
            for j in range(J):
                P = cat_prob(a[j], b[j], q_nodes)   # (K, n_quad)
                probs_all *= P[Y[i, j]]
            marginal = np.sum(q_weights * probs_all)
            ll += math.log(max(marginal, 1e-300))
        return -ll

    p0 = np.concatenate([np.zeros(J)] + [np.array([-1.0, 0.5, 0.5, 0.5][:Ktm]) for _ in range(J)])
    res = minimize(neg_ll, p0, method="L-BFGS-B")
    a, b = unpack(res.x)
    return {"a_discrimination": a, "b_thresholds": b,
            "log_lik": float(-res.fun), "K_categories": int(K),
            "method": "Samejima Graded Response Model (MML)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, J, K = 500, 5, 4    # 5 items, 4 ordered categories each
    theta_true = rng.normal(0, 1, n)
    a_true = rng.uniform(0.8, 1.8, J)
    # Thresholds -1, 0, 1 for each item (with per-item jitter)
    b_true = np.tile(np.array([-1.0, 0.0, 1.0]), (J, 1)) + rng.normal(0, 0.2, (J, K - 1))
    Y = np.zeros((n, J), dtype=int)
    for i in range(n):
        for j in range(J):
            P_cum = np.array([1.0] + [_sigmoid(a_true[j] * (theta_true[i] - b_true[j, m])) for m in range(K - 1)] + [0.0])
            probs = P_cum[:-1] - P_cum[1:]
            Y[i, j] = int(rng.choice(K, p=probs / probs.sum()))

    fit = grm_mml(Y, K=K, n_quad=12)
    print("=== Samejima GRM ===")
    print(f"  log-lik = {fit['log_lik']:.2f}")
    print(f"  correlation of a_hat with a_true = {np.corrcoef(fit['a_discrimination'], a_true)[0, 1]:.3f}")
    print(f"  correlation of b_hat.flatten with b_true.flatten = "
          f"{np.corrcoef(fit['b_thresholds'].ravel(), b_true.ravel())[0, 1]:.3f}")

    print("\n--- library cross-check (R mirt::mirt itemtype = 'graded') ---")

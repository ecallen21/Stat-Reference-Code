"""Partial Credit Model (Masters 1982) + Generalized PCM (Muraki 1992)
(Reference §22.8).

PCM: Rasch-family model for polytomous items with K ordered categories.
Category probability for k = 0, ..., K-1:

    P_j(k | theta) = exp( sum_{h=0}^{k} (theta - delta_jh) )
                     / sum_{k'=0}^{K-1} exp( sum_{h=0}^{k'} (theta - delta_jh) )

with delta_j0 = 0.  Each delta_jh is the "step difficulty" from h-1 to h.
Unlike GRM, PCM's step difficulties need not be ordered - reversals are
possible and interpretable (locally hardest transition).

GPCM (Muraki 1992) adds a discrimination a_j:
    numerator = exp(a_j sum_{h=0}^{k} (theta - delta_jh))

Marginal MLE via Gauss-Hermite; the demo implements GPCM.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def gpcm_mml(Y, K: int, n_quad: int = 12) -> dict:
    """Muraki GPCM Marginal MLE."""
    Y = np.asarray(Y, dtype=int); n, J = Y.shape
    q_nodes, q_weights = np.polynomial.hermite_e.hermegauss(n_quad)
    q_weights = q_weights / math.sqrt(2 * math.pi)
    Ktm = K - 1

    def unpack(params):
        a = np.exp(params[:J])
        delta = params[J:].reshape(J, Ktm)
        return a, delta

    def cat_prob(a_j, delta_j, theta):
        """Return (K x n_theta) category probabilities."""
        cumul = np.zeros((K, len(theta)))
        for k in range(1, K):
            cumul[k] = cumul[k - 1] + a_j * (theta - delta_j[k - 1])
        cumul -= cumul.max(axis=0, keepdims=True)
        e = np.exp(cumul)
        return e / e.sum(axis=0, keepdims=True)

    def neg_ll(params):
        a, delta = unpack(params)
        ll = 0.0
        for i in range(n):
            joint = np.ones(n_quad)
            for j in range(J):
                P = cat_prob(a[j], delta[j], q_nodes)  # (K, n_quad)
                joint *= P[Y[i, j]]
            marginal = np.sum(q_weights * joint)
            ll += math.log(max(marginal, 1e-300))
        return -ll

    p0 = np.concatenate([np.zeros(J)] + [np.linspace(-1, 1, Ktm) for _ in range(J)])
    res = minimize(neg_ll, p0, method="L-BFGS-B")
    a, delta = unpack(res.x)
    return {"a_discrimination": a, "delta_step_difficulty": delta,
            "log_lik": float(-res.fun),
            "method": "Muraki GPCM (MML + Gauss-Hermite)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, J, K = 500, 4, 4
    theta_true = rng.normal(0, 1, n)
    a_true = rng.uniform(0.8, 1.6, J)
    delta_true = np.tile(np.array([-1.0, 0.0, 1.0]), (J, 1)) + rng.normal(0, 0.2, (J, K - 1))
    # Generate GPCM
    Y = np.zeros((n, J), dtype=int)
    for i in range(n):
        for j in range(J):
            cumul = np.zeros(K)
            for k in range(1, K):
                cumul[k] = cumul[k - 1] + a_true[j] * (theta_true[i] - delta_true[j, k - 1])
            cumul -= cumul.max()
            probs = np.exp(cumul); probs /= probs.sum()
            Y[i, j] = int(rng.choice(K, p=probs))

    fit = gpcm_mml(Y, K=K, n_quad=12)
    print("=== GPCM MML ===")
    print(f"  log-lik = {fit['log_lik']:.2f}")
    print(f"  correlation of a_hat with a_true = {np.corrcoef(fit['a_discrimination'], a_true)[0, 1]:.3f}")
    print(f"  correlation of delta_hat with delta_true = "
          f"{np.corrcoef(fit['delta_step_difficulty'].ravel(), delta_true.ravel())[0, 1]:.3f}")

    print("\n--- library cross-check (R mirt::mirt itemtype = 'gpcm') ---")

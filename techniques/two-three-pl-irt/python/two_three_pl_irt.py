"""2PL and 3PL Item Response Theory (Reference §22.6).

2PL (Birnbaum 1968)
    Pr(y_ij = 1 | theta_i, a_j, b_j) = 1 / (1 + exp(-a_j (theta_i - b_j)))

    a_j : DISCRIMINATION (slope) - how sharply Pr rises around b_j
    b_j : DIFFICULTY

3PL (Birnbaum / Lord 1980)
    Pr(y_ij = 1 | ...) = c_j + (1 - c_j) * sigmoid(a_j (theta_i - b_j))
    c_j : PSEUDO-GUESSING lower asymptote (e.g. 4-option MC -> 0.25)

Marginal ML estimation
    Integrate person ability theta ~ N(0, 1) via Gauss-Hermite quadrature.
    Log-lik = sum_i log integral_theta prod_j Pr(y_ij | theta) phi(theta) d theta.
    Optimize (a, b, c) by BFGS.

The demo below implements 2PL MML with 15-point Gauss-Hermite quadrature.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def _sigmoid(x): return 1 / (1 + np.exp(-x))


def two_pl_mml(Y, n_quad: int = 15) -> dict:
    """2PL Marginal MLE via Gauss-Hermite quadrature."""
    Y = np.asarray(Y, dtype=float); n, J = Y.shape
    q_nodes, q_weights = np.polynomial.hermite_e.hermegauss(n_quad)
    # theta ~ N(0, 1); Gauss-Hermite gives nodes for exp(-x^2/2)/sqrt(2 pi)
    q_weights = q_weights / math.sqrt(2 * math.pi)
    def unpack(params):
        a = np.exp(params[:J])                       # a > 0 via exp reparam
        b = params[J:]
        return a, b
    def neg_ll(params):
        a, b = unpack(params)
        ll = 0.0
        for i in range(n):
            probs = _sigmoid(a[None, :] * (q_nodes[:, None] - b[None, :]))
            p_iq = np.prod(np.where(Y[i] == 1, probs, 1 - probs), axis=1)
            marginal = np.sum(q_weights * p_iq)
            ll += math.log(max(marginal, 1e-300))
        return -ll
    p0 = np.concatenate([np.zeros(J), np.zeros(J)])  # a_init = 1, b_init = 0
    res = minimize(neg_ll, p0, method="L-BFGS-B")
    a, b = unpack(res.x)
    return {"a_discrimination": a, "b_difficulty": b,
            "log_lik": float(-res.fun),
            "n_quad": int(n_quad),
            "method": "2PL Marginal MLE (Gauss-Hermite quadrature)"}


def _eap_theta(Y_i, a, b, n_quad: int = 21) -> float:
    """Expected a-posteriori theta estimate for one person."""
    q_nodes, q_weights = np.polynomial.hermite_e.hermegauss(n_quad)
    q_weights = q_weights / math.sqrt(2 * math.pi)
    probs = _sigmoid(a[None, :] * (q_nodes[:, None] - b[None, :]))
    p_iq = np.prod(np.where(Y_i == 1, probs, 1 - probs), axis=1)
    post = q_weights * p_iq
    return float(np.sum(q_nodes * post) / np.sum(post))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, J = 400, 12
    theta_true = rng.normal(0, 1, n)
    a_true = rng.uniform(0.7, 2.0, J)
    b_true = rng.uniform(-1.5, 1.5, J)
    P = _sigmoid(a_true[None, :] * (theta_true[:, None] - b_true[None, :]))
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    print("=== 2PL MML (Gauss-Hermite quadrature) ===")
    fit = two_pl_mml(Y, n_quad=15)
    print(f"  log-lik = {fit['log_lik']:.2f}")
    print(f"  correlation of a_hat with a_true = {np.corrcoef(fit['a_discrimination'], a_true)[0, 1]:.3f}")
    print(f"  correlation of b_hat with b_true = {np.corrcoef(fit['b_difficulty'], b_true)[0, 1]:.3f}")

    print("\n=== EAP theta estimates for 5 persons ===")
    for i in [0, 100, 200, 300, 399]:
        theta_est = _eap_theta(Y[i], fit["a_discrimination"], fit["b_difficulty"])
        print(f"  person {i}: theta_hat = {theta_est:6.3f}   true = {theta_true[i]:6.3f}")

    print("\n--- library cross-check (R ltm::ltm / mirt::mirt) ---")
    print("  R: ltm::ltm(Y ~ z1)  or  mirt::mirt(Y, 1, itemtype = '2PL')")

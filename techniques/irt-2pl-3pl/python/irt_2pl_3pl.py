"""2PL / 3PL IRT models (Reference Sec 20.4, 20.5).

Birnbaum 1968.  Item Response Theory extensions of the Rasch (1PL)
model:

  2PL   P(y_ij = 1 | theta_i) = 1 / (1 + exp(-a_j (theta_i - b_j)))
        a_j = item discrimination,  b_j = item difficulty

  3PL   P(y_ij = 1 | theta_i) = c_j + (1 - c_j) * 2PL_ij
        c_j = pseudo-guessing (lower asymptote), > 0

Estimated by marginal MLE (integrating out theta) or joint MLE.
Here we use a compact EM with theta at Gauss-Hermite nodes.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy.optimize import minimize


def _sigmoid(z): return 1 / (1 + np.exp(-z))


def item_probs_2pl(theta, a, b):
    """theta shape (K,), a/b shape (J,).  Returns (K, J)."""
    return _sigmoid(a[None, :] * (theta[:, None] - b[None, :]))


def item_probs_3pl(theta, a, b, c):
    p2 = item_probs_2pl(theta, a, b)
    return c[None, :] + (1 - c[None, :]) * p2


def marginal_ll(params, Y, K_quad=15, model="2pl"):
    n, J = Y.shape
    if model == "2pl":
        a = np.exp(params[:J]); b = params[J:2 * J]
        c = np.zeros(J)
    else:
        a = np.exp(params[:J]); b = params[J:2 * J]
        c = _sigmoid(params[2 * J:3 * J])       # keep in (0, 1)
    # Gauss-Hermite nodes for theta ~ N(0, 1)
    theta, w = np.polynomial.hermite_e.hermegauss(K_quad)
    w = w / np.sqrt(2 * np.pi)
    p = item_probs_3pl(theta, a, b, c) if model == "3pl" else item_probs_2pl(theta, a, b)
    log_p = np.log(p + 1e-12); log_1p = np.log(1 - p + 1e-12)
    ll = 0.0
    for i in range(n):
        log_lik_k = (Y[i] * log_p + (1 - Y[i]) * log_1p).sum(axis=1)
        ll += np.log((w * np.exp(log_lik_k - log_lik_k.max())).sum() + 1e-300) + log_lik_k.max()
    return -ll


def fit_irt(Y, model="2pl"):
    J = Y.shape[1]
    if model == "2pl":
        x0 = np.zeros(2 * J)
    else:
        x0 = np.zeros(3 * J)
    r = minimize(lambda p: marginal_ll(p, Y, model=model), x0=x0,
                 method="L-BFGS-B", options={"maxiter": 100})
    return r


if __name__ == "__main__":
    print("=== 2PL / 3PL IRT ===\n")
    rng = np.random.default_rng(0)
    n, J = 800, 6
    theta_true = rng.normal(0, 1, n)
    a_true = np.array([1.0, 1.2, 0.8, 1.5, 1.1, 0.9])
    b_true = np.linspace(-1.5, 1.5, J)
    c_true = 0.15                                # guessing 15% each item
    p = c_true + (1 - c_true) / (1 + np.exp(-a_true * (theta_true[:, None] - b_true)))
    Y = (rng.random((n, J)) < p).astype(int)

    r2 = fit_irt(Y, model="2pl")
    r3 = fit_irt(Y, model="3pl")
    a2 = np.exp(r2.x[:J]); b2 = r2.x[J:2 * J]
    a3 = np.exp(r3.x[:J]); b3 = r3.x[J:2 * J]
    c3 = 1 / (1 + np.exp(-r3.x[2 * J:3 * J]))
    print(f"  True a = {a_true}")
    print(f"  2PL a  = {a2.round(2)}")
    print(f"  3PL a  = {a3.round(2)}")
    print(f"  True b = {b_true.round(2)}")
    print(f"  2PL b  = {b2.round(2)}")
    print(f"  3PL b  = {b3.round(2)}")
    print(f"  True c = {c_true}   3PL c mean = {c3.mean():.3f}\n")

    print("--- library cross-check (R ltm, mirt, TAM; Python girth, py-irt) ---")

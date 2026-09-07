"""Deep survival networks (Reference Sec 11.27).

Katzman et al. 2018 'DeepSurv: personalized treatment recommender
system using a Cox proportional hazards deep neural network', BMC
Med Res Meth. A neural-network extension of Cox PH:

    h(t | x) = h_0(t) * exp( g_theta(x) )

with g_theta a NN. Loss = negative Cox partial likelihood:

    L(theta) = - sum_{i: e_i = 1}  [ g_theta(x_i)
                                     - log sum_{j in R(t_i)} exp(g_theta(x_j)) ]

Related architectures:
    * DeepHit          -- discrete-time competing risks (Lee et al. 2018)
    * Nnet-survival    -- discrete-time hazard head
    * Deep Cox time-varying -- pyramid architectures
    * Neural ODE Survival   -- continuous-time via neural ODE

We implement a compact 1-hidden-layer 'DeepSurv' from scratch with
manual gradient computation of the partial likelihood.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def sigmoid(x): return 1 / (1 + np.exp(-x))


def cox_pl_gradient(g, t, e):
    """Gradient of NEGATIVE partial likelihood wrt g (n-vector)."""
    order = np.argsort(-t)     # descending time
    t = t[order]; e = e[order]
    exp_g = np.exp(g[order])
    #  Descending sort: at position k, risk set R(t_k) = entries 0..k.
    cum = np.cumsum(exp_g)     # risk-set sum at each event time
    #  Gradient contribution: for entry i, sum over events k with i in R(t_k).
    #  With descending sort, i in R(t_k) iff i <= k, so we need suffix cumsum.
    increments = np.where(e == 1, 1.0 / cum, 0.0)
    suffix = np.cumsum(increments[::-1])[::-1]     # suffix[i] = sum_{k >= i} 1/cum[k]
    grad_sorted = -e + exp_g * suffix
    #  Undo the sort
    grad = np.zeros_like(g)
    grad[order] = grad_sorted
    return grad


def deep_surv_fit(X, t, e, hidden=8, lr=0.05, epochs=200, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    W1 = rng.normal(scale=0.3, size=(p, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(scale=0.3, size=(hidden, 1))
    b2 = 0.0

    for ep in range(epochs):
        Z = np.tanh(X @ W1 + b1)
        g = (Z @ W2).ravel() + b2
        grad_g = cox_pl_gradient(g, t, e)
        #  Backprop
        dW2 = Z.T @ grad_g.reshape(-1, 1)
        dZ = grad_g.reshape(-1, 1) @ W2.T * (1 - Z ** 2)
        dW1 = X.T @ dZ
        db1 = dZ.sum(axis=0)
        W1 -= lr * dW1 / n
        b1 -= lr * db1 / n
        W2 -= lr * dW2 / n
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def deep_surv_predict_risk(model, X):
    Z = np.tanh(X @ model["W1"] + model["b1"])
    return (Z @ model["W2"]).ravel()


def cindex(risk, t, e):
    concord = 0; total = 0
    for i in range(len(t)):
        if e[i] == 0: continue
        cmp_t = t > t[i]
        total += cmp_t.sum()
        concord += (risk[i] > risk[cmp_t]).sum() + 0.5 * (risk[i] == risk[cmp_t]).sum()
    return concord / total if total > 0 else 0.5


if __name__ == "__main__":
    print("=== Deep survival network (DeepSurv-flavour) ===\n")
    rng = np.random.default_rng(0)
    n = 500; p = 4
    X = rng.normal(size=(n, p))
    #  Nonlinear true log hazard: 0.5 * x0 + 0.7 * x1^2 - 0.4 * x2 * x3
    lp = 0.5 * X[:, 0] + 0.7 * X[:, 1] ** 2 - 0.4 * X[:, 2] * X[:, 3]
    t_true = rng.weibull(1.5, size=n) * np.exp(-lp)
    c = rng.exponential(3.0, size=n)
    e = (t_true <= c).astype(int)
    t = np.minimum(t_true, c)

    model = deep_surv_fit(X, t, e, hidden=8, lr=0.1, epochs=300)
    risk_nn = deep_surv_predict_risk(model, X)

    #  Cox baseline: linear index only
    def linear_cox(X, t, e):
        beta = np.zeros(X.shape[1])
        for _ in range(200):
            g = X @ beta
            grad = X.T @ cox_pl_gradient(g, t, e)
            beta -= 0.05 * grad / len(t)
        return beta
    beta_lin = linear_cox(X, t, e)
    risk_lin = X @ beta_lin

    print(f"  n = {n}, event rate = {e.mean() * 100:.1f}%")
    print(f"  Harrell C:")
    print(f"    linear Cox        = {cindex(risk_lin, t, e):.3f}")
    print(f"    DeepSurv (H = 8)  = {cindex(risk_nn, t, e):.3f}")

    print("\n--- library cross-check (survivalmodels / pycox Python; ellmer / R6 R) ---")

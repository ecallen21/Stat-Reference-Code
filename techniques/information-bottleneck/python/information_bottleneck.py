"""Information bottleneck (Reference Sec 34.12).

Tishby, Pereira & Bialek (1999) 'The information bottleneck method.'

Given input X and target Y, learn a compressed representation T that
MAXIMISES  I(T; Y)  while KEEPING  I(T; X)  small:

  L_IB  =  I(T; X)  -  beta * I(T; Y).

Beta trades COMPRESSION (small I(T; X)) against PREDICTION (large
I(T; Y)). Iterative Blahut-Arimoto updates:

  p(t | x) = p(t)/Z * exp(-beta * KL( p(y|x) || p(y|t) )).

Related: information bottleneck theory of deep learning (Tishby-Zaslavsky 2015).

Here we implement discrete IB on synthetic (X, Y) with cluster
structure and sweep beta to trace the information-plane.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def mutual_info(p_xy):
    p_x = p_xy.sum(axis=1); p_y = p_xy.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_xy / (p_x[:, None] * p_y[None, :] + 1e-12)
        return float(np.nansum(p_xy * np.log(ratio + 1e-12)))


def ib_discrete(p_xy, T=5, beta=1.0, max_iter=100, seed=0):
    """Discrete IB via Blahut-Arimoto iteration."""
    rng = np.random.default_rng(seed)
    n_x, n_y = p_xy.shape
    p_x = p_xy.sum(axis=1)
    p_y_given_x = p_xy / (p_x[:, None] + 1e-12)
    # Init p(t | x)
    p_t_given_x = rng.random((n_x, T))
    p_t_given_x = p_t_given_x / p_t_given_x.sum(axis=1, keepdims=True)
    for _ in range(max_iter):
        # p(t)
        p_t = (p_x[:, None] * p_t_given_x).sum(axis=0)
        # p(y | t) = sum_x p(y | x) p(x | t) = sum_x p(y | x) p(t | x) p(x) / p(t)
        num = np.einsum("xy,xt,x->ty", p_y_given_x, p_t_given_x, p_x)
        p_y_given_t = num / (p_t[:, None] + 1e-12)
        # Update p(t | x) proportional to p(t) exp(-beta KL(p(y|x) || p(y|t)))
        KL_xt = np.zeros((n_x, T))
        for x in range(n_x):
            for t in range(T):
                py_x = p_y_given_x[x] + 1e-12; py_t = p_y_given_t[t] + 1e-12
                KL_xt[x, t] = float(np.sum(py_x * np.log(py_x / py_t)))
        log_num = np.log(p_t + 1e-12)[None, :] - beta * KL_xt
        log_num -= log_num.max(axis=1, keepdims=True)
        p_t_given_x_new = np.exp(log_num)
        p_t_given_x_new /= p_t_given_x_new.sum(axis=1, keepdims=True)
        if np.max(np.abs(p_t_given_x_new - p_t_given_x)) < 1e-6:
            p_t_given_x = p_t_given_x_new; break
        p_t_given_x = p_t_given_x_new
    # Compute I(T; X) and I(T; Y)
    p_tx = p_t_given_x * p_x[:, None]                # (n_x, T)
    p_ty = np.einsum("xt,xy->ty", p_t_given_x, p_xy)  # (T, n_y)
    I_TX = mutual_info(p_tx.T)                        # transpose to (T, n_x)
    I_TY = mutual_info(p_ty)
    return I_TX, I_TY


if __name__ == "__main__":
    print("=== Information bottleneck (Tishby 1999) ===\n")
    rng = np.random.default_rng(0)
    # Synthetic X, Y with 3-cluster joint distribution
    n_x, n_y = 12, 3
    p_xy = np.zeros((n_x, n_y))
    for x in range(n_x):
        y = x // 4
        p_xy[x, y] = 1
    p_xy = p_xy + 0.1 * rng.random((n_x, n_y))
    p_xy = p_xy / p_xy.sum()
    print(f"  I(X; Y) full = {mutual_info(p_xy):.4f} nats")

    print(f"\n  {'beta':>6}  {'I(T; X)':>9}  {'I(T; Y)':>9}")
    for beta in (0.1, 0.5, 1.0, 3.0, 10.0):
        I_TX, I_TY = ib_discrete(p_xy, T=5, beta=beta, max_iter=100)
        print(f"  {beta:>6.1f}  {I_TX:>9.4f}  {I_TY:>9.4f}")

    print("\n  As beta grows, I(T; X) grows (T retains more input info) and I(T; Y) approaches I(X; Y).\n")
    print("--- library cross-check (Python information-bottleneck; JIDT; deepIB reference) ---")

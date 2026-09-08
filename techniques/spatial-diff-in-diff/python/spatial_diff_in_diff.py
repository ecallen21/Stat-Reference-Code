"""Spatial difference-in-differences (Reference Sec 15.43).

Delgado & Florax 2015; Chagas et al. 2016. Extends DiD to spatial
data where treatment ASSIGNMENT spills over across neighboring units:

    y_it = alpha + beta * D_i + gamma * Post_t + delta * D_i * Post_t
           + rho * W * y_it + eps_it

with W a row-normalised spatial weights matrix. Alternatively the
'SLX-DiD' just adds spatial lag of the treatment:

    y_it = alpha + beta * D_i + gamma * Post_t + delta * D_i * Post_t
           + theta * (W * D_i) * Post_t + eps_it

The DIRECT effect is delta; the SPILLOVER effect is theta * (W * D)
on untreated neighbors of treated units.

We simulate a 2-arm setup on a 1-D lattice where treatment SPILLS to
neighbors and estimate direct + spillover coefficients.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def slx_did(y, D, Post, W):
    """Two-period SLX-DiD: y = alpha + b*D + c*Post + d*D*Post + theta*(W D)*Post + eps."""
    n = len(y)
    X = np.column_stack([np.ones(n), D, Post, D * Post, (W @ D) * Post])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"alpha": beta[0], "beta_D": beta[1], "gamma_Post": beta[2],
            "delta_direct": beta[3], "theta_spillover": beta[4]}


if __name__ == "__main__":
    print("=== Spatial diff-in-differences (SLX-DiD) ===\n")
    rng = np.random.default_rng(0)
    N = 40                        # 40 spatial units in a line
    T = 2                          # pre + post
    #  Treatment on units 15..24 (contiguous cluster)
    treated_mask = np.zeros(N)
    treated_mask[15:25] = 1
    #  Row-normalised weights matrix (nearest-neighbour on the line)
    W = np.zeros((N, N))
    for i in range(N):
        if i > 0:     W[i, i - 1] = 1
        if i < N - 1: W[i, i + 1] = 1
        s = W[i].sum()
        if s > 0: W[i] /= s

    #  True direct effect = 2.0, spillover to neighbours = 0.8
    delta_true = 2.0; theta_true = 0.8
    D_it = []; Post_it = []; y_it = []
    for i in range(N):
        for t in range(T):
            D_it.append(treated_mask[i])
            Post_it.append(t)
            spillover = float((W @ treated_mask)[i]) * theta_true * t
            direct = treated_mask[i] * delta_true * t
            y = 5 + 0.3 * i + 0.5 * t + direct + spillover + rng.normal(scale=0.5)
            y_it.append(y)
    D_it = np.array(D_it); Post_it = np.array(Post_it); y_it = np.array(y_it)

    #  For panel SLX-DiD, W must map unit-level to obs-level
    #  Simplest: treat each obs as its own -> compute W_obs
    W_obs = np.zeros((N * T, N * T))
    for i in range(N):
        for t in range(T):
            row = i * T + t
            for j in range(N):
                col = j * T + t
                W_obs[row, col] = W[i, j]

    r = slx_did(y_it, D_it, Post_it, W_obs)
    print(f"  True direct effect (delta)     = {delta_true:.2f}")
    print(f"  True spillover effect (theta)  = {theta_true:.2f}")
    print(f"  Estimated delta                 = {r['delta_direct']:+.3f}")
    print(f"  Estimated theta                 = {r['theta_spillover']:+.3f}")

    print("\n--- library cross-check (spatialreg / plm R; PySAL / linearmodels Python) ---")

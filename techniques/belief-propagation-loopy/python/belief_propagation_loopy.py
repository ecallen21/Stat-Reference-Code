"""Loopy belief propagation (Reference Sec 47.121).

Pearl 1988 'Probabilistic Reasoning in Intelligent Systems' (tree
BP); Weiss 1997 / Yedidia-Freeman-Weiss 2005 (loopy variant).
Sum-product on a pairwise Markov random field:

    m_{i -> j}^{new}(x_j) proportional to
        sum_{x_i} phi_i(x_i) psi_{ij}(x_i, x_j)
                 prod_{k in N(i) \ j} m_{k -> i}^{old}(x_i)

Beliefs at node i:  b_i(x_i) proportional to  phi_i(x_i) * prod_{k in N(i)} m_{k -> i}(x_i).

Exact on trees, approximate ("loopy") on graphs with cycles.
Demonstrated on a small Ising grid with attractive coupling.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def loopy_bp_ising(H, J, T=1.0, damping=0.5, max_iter=200, tol=1e-6):
    """Loopy BP on a pairwise Ising: nodes s_i in {-1, +1},
    phi_i(s) = exp(H_i s / T),  psi_ij(s_i, s_j) = exp(J_ij s_i s_j / T).
    H : (n,) node fields, J : (n, n) symmetric coupling matrix.
    """
    n = len(H)
    nbrs = [np.where(J[i] != 0)[0] for i in range(n)]
    # Message m[i, j] over states of j (2,)
    m = np.ones((n, n, 2))
    for _ in range(max_iter):
        diff = 0.0
        m_new = np.ones_like(m)
        for i in range(n):
            for j in nbrs[i]:
                for s_j in (0, 1):        # 0 -> -1, 1 -> +1
                    s_j_val = -1 if s_j == 0 else 1
                    tot = 0.0
                    for s_i in (0, 1):
                        s_i_val = -1 if s_i == 0 else 1
                        phi = np.exp(H[i] * s_i_val / T)
                        psi = np.exp(J[i, j] * s_i_val * s_j_val / T)
                        prod = 1.0
                        for k in nbrs[i]:
                            if k == j: continue
                            prod *= m[k, i, s_i]
                        tot += phi * psi * prod
                    m_new[i, j, s_j] = tot
                # Normalise
                s = m_new[i, j].sum()
                if s > 0: m_new[i, j] /= s
            diff = max(diff, float(np.max(np.abs(m_new[i] - m[i]))))
        m = damping * m + (1 - damping) * m_new
        if diff < tol:
            break

    # Beliefs
    beliefs = np.zeros((n, 2))
    for i in range(n):
        for s_i in (0, 1):
            s_val = -1 if s_i == 0 else 1
            phi = np.exp(H[i] * s_val / T)
            prod = 1.0
            for k in nbrs[i]:
                prod *= m[k, i, s_i]
            beliefs[i, s_i] = phi * prod
        beliefs[i] /= beliefs[i].sum()
    return beliefs


if __name__ == "__main__":
    print("=== Loopy belief propagation (Pearl 1988; Yedidia et al 2005) ===\n")
    n_side = 3
    n = n_side * n_side
    # 3x3 grid, ferromagnetic coupling J=+1, no external field
    J = np.zeros((n, n))
    for i in range(n_side):
        for j in range(n_side):
            u = i * n_side + j
            for di, dj in [(0, 1), (1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < n_side and 0 <= nj < n_side:
                    v = ni * n_side + nj
                    J[u, v] = 1.0; J[v, u] = 1.0
    # Small positive field on centre node
    H = np.zeros(n); H[4] = 0.3

    beliefs = loopy_bp_ising(H, J, T=2.5)
    print("  Beliefs P(s = +1) on 3x3 Ising grid (field 0.3 on centre):")
    print(np.round(beliefs[:, 1].reshape(n_side, n_side), 3))

    print("\n  Positive field on centre biases the whole grid via BP messages.")

    # Sanity check via brute-force enumeration
    P_true = np.zeros((n, 2))
    Z = 0.0
    for k in range(1 << n):
        s = np.array([1 if (k >> i) & 1 else -1 for i in range(n)])
        E = -H @ s - 0.5 * s @ J @ s
        p = np.exp(-E / 2.5)
        Z += p
        for i in range(n):
            P_true[i, 1 if s[i] == 1 else 0] += p
    P_true /= Z
    print("\n  Exact enumeration P(s = +1):")
    print(np.round(P_true[:, 1].reshape(n_side, n_side), 3))

    print("\n--- library cross-check (bnlearn / gRain R; pgmpy / libDAI Python) ---")

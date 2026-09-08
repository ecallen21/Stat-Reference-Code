"""Tensor decomposition -- Tucker & CP (Reference Sec 6.17).

Tucker 1966, Harshman 1970 (CANDECOMP / PARAFAC). Extend SVD to
higher-order tensors X in R^{I x J x K}.

CP (canonical polyadic):
    X approx sum_r a_r  outer_prod  b_r  outer_prod  c_r     (rank R)
    Parameters: R * (I + J + K).

Tucker (higher-order SVD):
    X approx G  x_1 U^{(1)}  x_2 U^{(2)}  x_3 U^{(3)}
    Core G in R^{R1 x R2 x R3}, factor matrices U^{(m)}.

Widely used in chemometrics, recommender systems, neuroimaging,
multi-way ANOVA.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def unfold(X, mode):
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def khatri_rao(A, B):
    ai, R = A.shape; bi = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(ai * bi, R)


def cp_als(X, rank=3, max_iter=100, tol=1e-6, seed=0):
    """CP via alternating least squares."""
    rng = np.random.default_rng(seed)
    shp = X.shape; d = X.ndim
    factors = [rng.normal(size=(shp[m], rank)) for m in range(d)]
    for it in range(max_iter):
        for m in range(d):
            others = [f for j, f in enumerate(factors) if j != m]
            #  Reverse order for khatri-rao alignment
            kr = others[-1]
            for k in range(len(others) - 2, -1, -1):
                kr = khatri_rao(others[k], kr)
            V = np.ones((rank, rank))
            for f in others:
                V *= f.T @ f
            factors[m] = unfold(X, m) @ kr @ np.linalg.pinv(V)
    #  Reconstruct
    X_hat = np.einsum("ir,jr,kr->ijk", *factors)
    return {"factors": factors, "reconstruction": X_hat,
            "err": float(np.linalg.norm(X - X_hat) / np.linalg.norm(X))}


def n_mode_product(T, U, mode):
    """Multiply tensor T by matrix U along `mode`."""
    T_unf = np.moveaxis(T, mode, 0)
    shp = T_unf.shape
    out = (U @ T_unf.reshape(shp[0], -1)).reshape(U.shape[0], *shp[1:])
    return np.moveaxis(out, 0, mode)


def hosvd(X, ranks):
    """Higher-order SVD (Tucker approx via mode-wise SVD)."""
    factors = []
    for m in range(X.ndim):
        u, _, _ = np.linalg.svd(unfold(X, m), full_matrices=False)
        factors.append(u[:, :ranks[m]])
    G = X.copy()
    for m in range(X.ndim):
        G = n_mode_product(G, factors[m].T, m)
    return {"factors": factors, "core": G}


if __name__ == "__main__":
    print("=== Tensor decomposition: CP-ALS + HOSVD (Tucker) ===\n")
    rng = np.random.default_rng(0)
    #  Build a rank-2 tensor + noise
    I, J, K = 8, 6, 5; R = 2
    A = rng.normal(size=(I, R))
    B = rng.normal(size=(J, R))
    C = rng.normal(size=(K, R))
    X = np.einsum("ir,jr,kr->ijk", A, B, C) + 0.1 * rng.normal(size=(I, J, K))

    r = cp_als(X, rank=R, max_iter=200)
    print(f"  CP rank = {R}   reconstruction relative error = {r['err']:.4f}")

    r_over = cp_als(X, rank=4, max_iter=200)
    print(f"  CP rank = 4 (over-specified)     error = {r_over['err']:.4f}")

    t = hosvd(X, ranks=(3, 3, 3))
    X_hat_tucker = np.einsum("abc,ia,jb,kc->ijk", t["core"], *t["factors"])
    err_tucker = float(np.linalg.norm(X - X_hat_tucker) / np.linalg.norm(X))
    print(f"  Tucker (3, 3, 3) HOSVD         error = {err_tucker:.4f}")

    print("\n--- library cross-check (rTensor / multiway R; tensorly Python) ---")

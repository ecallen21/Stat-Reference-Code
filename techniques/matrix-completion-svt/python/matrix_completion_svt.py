"""Matrix completion via Singular Value Thresholding (Sec 47.43).

Cai, Candes & Shen 2010 'A singular value thresholding algorithm for
matrix completion', SIAM J Optim 20(4). Given entries M_ij observed
on index set Omega, recover a low-rank M by solving

    min ||X||_*   s.t.  P_Omega(X) = P_Omega(M)

where ||.||_* is nuclear norm. SVT iterates

    Y_{k+1} = Y_k + delta * P_Omega(M - X_k)
    X_{k+1} = D_tau(Y_{k+1})              (soft-threshold singular values)

with soft-threshold D_tau(A) = U diag((sigma - tau)_+) V'.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def svt_step(Y, tau):
    U, s, Vt = np.linalg.svd(Y, full_matrices=False)
    s_thr = np.maximum(s - tau, 0)
    return (U * s_thr) @ Vt, int((s_thr > 0).sum())


def matrix_completion_svt(M_obs, mask, tau, delta=None, max_iter=500, tol=1e-4):
    """SVT matrix completion. `mask` is bool array marking observed entries."""
    delta = delta or 1.2 * np.prod(M_obs.shape) / max(mask.sum(), 1)
    Y = np.zeros_like(M_obs, dtype=float)
    X = np.zeros_like(M_obs, dtype=float)
    for k in range(max_iter):
        X, rank = svt_step(Y, tau)
        resid = mask * (M_obs - X)
        Y = Y + delta * resid
        err = np.linalg.norm(resid) / max(np.linalg.norm(mask * M_obs), 1e-12)
        if err < tol:
            break
    return {"X": X, "rank": rank, "iters": k + 1, "resid_rel": float(err)}


if __name__ == "__main__":
    print("=== Matrix completion via SVT (Cai-Candes-Shen 2010) ===\n")
    rng = np.random.default_rng(0)
    n, m, r = 50, 40, 3
    U = rng.normal(size=(n, r)); V = rng.normal(size=(m, r))
    M = U @ V.T
    frac = 0.6
    mask = rng.uniform(size=M.shape) < frac
    M_obs = M * mask
    print(f"  Truth: {n}x{m}, rank {r}. Observed fraction = {mask.mean():.2f}"
          f" ({mask.sum()} entries).")

    tau = 5 * np.sqrt(n * m)    # Cai-Candes-Shen heuristic
    fit = matrix_completion_svt(M_obs, mask, tau=tau, delta=1.2 / frac,
                                max_iter=2000, tol=1e-5)
    rel = np.linalg.norm(fit["X"] - M) / np.linalg.norm(M)
    print(f"  Recovered rank  = {fit['rank']}  (truth {r})")
    print(f"  Iterations     = {fit['iters']}")
    print(f"  Relative error ||X - M||/||M|| = {rel:.4f}")

    print("\n--- library cross-check (softImpute R; fancyimpute Python) ---")

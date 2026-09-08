"""MMD two-sample test (Reference Sec 47.65).

Gretton, Borgwardt, Rasch, Scholkopf & Smola 2012 'A kernel two-
sample test', JMLR 13. Maximum Mean Discrepancy in a
characteristic RKHS:

    MMD^2(P, Q) = || mu_P - mu_Q ||_H^2
                 = E[k(x, x')] + E[k(y, y')] - 2 E[k(x, y)]

Unbiased empirical estimator (Gretton eq 3):

    MMD_u^2 = 1/(m(m-1)) sum_{i != j} k(x_i, x_j)
            + 1/(n(n-1)) sum_{i != j} k(y_i, y_j)
            - 2/(mn) sum_i sum_j k(x_i, y_j).

For characteristic kernels (RBF) MMD = 0 iff P = Q. Permutation
p-value under H_0: P = Q by pooling and re-splitting.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def rbf_kernel(X, Y, sigma):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-sq / (2 * sigma ** 2))


def _median_heuristic(X, Y):
    Z = np.vstack([X, Y])
    sq = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.sqrt(0.5 * np.median(sq[sq > 0]))


def mmd2_unbiased(X, Y, sigma=None):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    sigma = sigma or _median_heuristic(X, Y)
    m, n = len(X), len(Y)
    Kxx = rbf_kernel(X, X, sigma); np.fill_diagonal(Kxx, 0)
    Kyy = rbf_kernel(Y, Y, sigma); np.fill_diagonal(Kyy, 0)
    Kxy = rbf_kernel(X, Y, sigma)
    mmd2 = Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.sum() / (m * n)
    return float(mmd2), sigma


def mmd_permutation_test(X, Y, n_perm=500, sigma=None, rng=None):
    rng = rng or np.random.default_rng(0)
    obs, sigma = mmd2_unbiased(X, Y, sigma)
    Z = np.vstack([np.atleast_2d(X).reshape(len(X), -1),
                    np.atleast_2d(Y).reshape(len(Y), -1)])
    m = len(X); count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        Xp = Z[idx[:m]]; Yp = Z[idx[m:]]
        m2, _ = mmd2_unbiased(Xp, Yp, sigma)
        if m2 >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"mmd2": obs, "sigma": float(sigma), "p": float(p), "n_perm": n_perm}


if __name__ == "__main__":
    print("=== MMD two-sample test (Gretton et al 2012) ===\n")
    rng = np.random.default_rng(0)
    m = n = 200

    scenarios = [
        ("same N(0,1)         ", rng.normal(size=(m, 2)), rng.normal(size=(n, 2))),
        ("mean shift +0.3     ", rng.normal(size=(m, 2)), rng.normal(size=(n, 2)) + 0.3),
        ("variance shift 1.5x ", rng.normal(size=(m, 2)), 1.5 * rng.normal(size=(n, 2))),
        ("mixture shift       ", rng.normal(size=(m, 2)),
         np.vstack([rng.normal(loc=-1, size=(n // 2, 2)), rng.normal(loc=1, size=(n // 2, 2))])),
    ]
    for name, X, Y in scenarios:
        r = mmd_permutation_test(X, Y, n_perm=200)
        print(f"  {name}  MMD^2 = {r['mmd2']:+.5f}   p (perm) = {r['p']:.4f}   "
              f"sigma = {r['sigma']:.2f}")

    print("\n  Detects mean, variance and MIXTURE shifts that univariate")
    print("  tests (t, F, KS on marginals) can miss.")
    print("\n--- library cross-check (kernlab / eummd R; hyppo / torch-two-sample Python) ---")

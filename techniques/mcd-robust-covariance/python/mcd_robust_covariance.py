"""MCD — Minimum Covariance Determinant (Rousseeuw 1985;
Rousseeuw-Van Driessen 1999 FAST-MCD).

Find the h-subset (of n) with the SMALLEST covariance
determinant. Result: robust location vector T and covariance
matrix C with 50% breakdown point.

FAST-MCD algorithm:
    (1) many random p+1 point starts, form (mu, S)
    (2) C-step: keep h points with smallest Mahalanobis distance
                to (mu, S), recompute (mu, S)
    (3) iterate to convergence; pick best determinant.
"""

import numpy as np    # arrays + linalg


def mahalanobis(X, mu, S_inv):
    dx = X - mu
    return np.sum((dx @ S_inv) * dx, axis=1)


def c_step_mcd(X, mu, S, h):
    S_inv = np.linalg.pinv(S)
    d2 = mahalanobis(X, mu, S_inv)
    idx = np.argsort(d2)[:h]
    Xh = X[idx]
    mu_new = Xh.mean(axis=0)
    S_new = np.cov(Xh.T, bias=True)
    det = np.linalg.det(S_new)
    return mu_new, S_new, det, idx


def fast_mcd(X, alpha=0.75, n_starts=200, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    h = int(np.floor(alpha * n))
    top = []
    for _ in range(n_starts):
        idx0 = rng.choice(n, size=p + 1, replace=False)
        Xs = X[idx0]
        mu = Xs.mean(axis=0)
        S = np.cov(Xs.T, bias=True) + 1e-6 * np.eye(p)
        for _ in range(3):
            mu, S, det, _ = c_step_mcd(X, mu, S, h)
        top.append((det, mu, S))
    top.sort(key=lambda t: t[0])
    best = (np.inf, None, None)
    for det0, mu0, S0 in top[:10]:
        mu, S = mu0, S0
        prev = np.inf
        for _ in range(50):
            mu, S, det, _ = c_step_mcd(X, mu, S, h)
            if abs(prev - det) < 1e-12:
                break
            prev = det
        if det < best[0]:
            best = (det, mu, S)
    return best[1], best[2]


def demo():
    print("=== MCD — Minimum Covariance Determinant (Rousseeuw 1985) ===")
    rng = np.random.default_rng(2026)
    n, p = 200, 3
    mu_true = np.array([1.0, 2.0, 3.0])
    A = rng.standard_normal((p, p))
    Sigma_true = A @ A.T + np.eye(p)
    X_clean = rng.multivariate_normal(mu_true, Sigma_true, size=int(0.7 * n))
    X_out = rng.uniform(-10, 20, size=(int(0.3 * n), p))
    X = np.vstack([X_clean, X_out])
    rng.shuffle(X)

    mu_ml = X.mean(axis=0)
    S_ml = np.cov(X.T, bias=False)
    mu_mcd, S_mcd = fast_mcd(X, alpha=0.75, n_starts=200)

    err_ml = np.linalg.norm(mu_ml - mu_true)
    err_mcd = np.linalg.norm(mu_mcd - mu_true)
    S_err_ml = np.linalg.norm(S_ml - Sigma_true, "fro") / np.linalg.norm(Sigma_true, "fro")
    S_err_mcd = np.linalg.norm(S_mcd - Sigma_true, "fro") / np.linalg.norm(Sigma_true, "fro")
    print(f"  n = {n}, p = {p}, 30% wild outliers")
    print(f"  Sample mean err   MLE  = {err_ml:.3f}  MCD = {err_mcd:.3f}")
    print(f"  Sample cov  err   MLE  = {S_err_ml:.3f}  MCD = {S_err_mcd:.3f}")
    print(f"  True mu = {mu_true},   MCD mu = {np.round(mu_mcd, 3)}")


if __name__ == "__main__":
    demo()

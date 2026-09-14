"""Douglas-Rachford splitting (Douglas-Rachford 1956;
Combettes-Wajs 2005).

Solve min_x f(x) + g(x) using proximal operators of f and g:

    y = prox_{lambda f}(z)
    z = z + prox_{lambda g}(2 y - z) - y

Fixed points of the DR iteration lie in the minimiser set.
General splitting technique — includes ADMM as a special case.
Works when only PROX of each summand is available.
"""

import numpy as np    # arrays + linalg


def prox_l1(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)


def prox_quad(z, A, b, lam):
    """prox for 0.5 || A x - b ||^2: solve (I + lam A.T A) x = z + lam A.T b."""
    n = A.shape[1]
    return np.linalg.solve(np.eye(n) + lam * A.T @ A, z + lam * A.T @ b)


def douglas_rachford(prox_f, prox_g, x0, lam=1.0, max_iter=200, tol=1e-9):
    z = np.array(x0, dtype=float)
    history = []
    for k in range(max_iter):
        y = prox_f(z, lam)
        z_new = z + prox_g(2 * y - z, lam) - y
        if np.linalg.norm(z_new - z) < tol:
            z = z_new
            break
        z = z_new
        history.append(np.linalg.norm(y))
    x = prox_f(z, lam)
    return x, history


def demo():
    print("=== Douglas-Rachford splitting (Douglas-Rachford 1956) ===")
    rng = np.random.default_rng(2026)
    n, d = 200, 100
    A = rng.standard_normal((n, d))
    x_true = np.zeros(d)
    x_true[:10] = rng.standard_normal(10) * 2
    b = A @ x_true + 0.1 * rng.standard_normal(n)

    lam_l1 = 0.5

    # LASSO: min 0.5 ||A x - b||^2 + lam ||x||_1
    def prox_f(z, lam):    # least-squares part
        return prox_quad(z, A, b, lam)

    def prox_g(z, lam):    # L1 part
        return prox_l1(z, lam * lam_l1)

    x_dr, hist = douglas_rachford(prox_f, prox_g, np.zeros(d), lam=0.1,
                                   max_iter=500)

    # compare with sklearn Lasso via alpha = lam_l1 / n
    try:
        from sklearn.linear_model import Lasso
        clf = Lasso(alpha=lam_l1 / n, fit_intercept=False, max_iter=5000)
        clf.fit(A, b)
        x_sk = clf.coef_
    except ImportError:
        x_sk = np.full(d, np.nan)

    def obj(x):
        return 0.5 * np.sum((A @ x - b) ** 2) + lam_l1 * np.sum(np.abs(x))

    print(f"  DR nonzero support = {int((np.abs(x_dr) > 1e-3).sum())}, "
          f"objective = {obj(x_dr):.4f}")
    if not np.isnan(x_sk[0]):
        print(f"  Lasso (sklearn) support = {int((np.abs(x_sk) > 1e-3).sum())}, "
              f"objective = {obj(x_sk):.4f}")
    print(f"  True nonzero support = 10")


if __name__ == "__main__":
    demo()

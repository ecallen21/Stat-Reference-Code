"""Least Trimmed Squares (Rousseeuw 1984 JASA).

Robust regression minimising the sum of the SMALLEST h squared
residuals:

    min_beta sum_{i=1..h} (r_(i)(beta))^2

where r_(i) is the i-th smallest squared residual. Breakdown
point up to 50% (much higher than OLS's 0%) when h = ceil((n+p+1)/2).

Standard FAST-LTS algorithm (Rousseeuw-Van Driessen 2006):
    (1) random start (elemental) subsets, run 3 C-steps each
    (2) pick top 10 subsets, iterate C-steps to convergence
    (3) return best fit
"""

import numpy as np    # arrays + linalg


def c_step(X, y, beta, h):
    """One 'concentration' step: refit on the h smallest-residual points."""
    r2 = (y - X @ beta) ** 2
    idx = np.argsort(r2)[:h]
    Xh, yh = X[idx], y[idx]
    beta_new = np.linalg.lstsq(Xh, yh, rcond=None)[0]
    obj = np.sum(np.sort(r2)[:h])
    return beta_new, obj, idx


def fast_lts(X, y, alpha=0.75, n_starts=200, seed=0):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    h = int(np.floor(alpha * n))
    best_obj = np.inf
    best_beta = None
    # phase 1: many random starts, 3 C-steps
    top10 = []
    for _ in range(n_starts):
        idx0 = rng.choice(n, size=p, replace=False)
        try:
            beta = np.linalg.lstsq(X[idx0], y[idx0], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        for _ in range(3):
            beta, obj, _ = c_step(X, y, beta, h)
        top10.append((obj, beta))
    top10.sort(key=lambda t: t[0])
    for obj0, beta0 in top10[:10]:
        beta = beta0
        prev_obj = np.inf
        for _ in range(50):
            beta, obj, _ = c_step(X, y, beta, h)
            if abs(prev_obj - obj) < 1e-10:
                break
            prev_obj = obj
        if obj < best_obj:
            best_obj = obj
            best_beta = beta
    return best_beta, best_obj


def demo():
    print("=== Least Trimmed Squares (Rousseeuw 1984 JASA) ===")
    rng = np.random.default_rng(2026)
    n, p = 100, 3
    beta_true = np.array([1.0, -2.0, 3.0])
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
    y = X @ beta_true + rng.standard_normal(n) * 0.5
    # inject 30 wild outliers
    outlier_idx = rng.choice(n, size=30, replace=False)
    y[outlier_idx] = y[outlier_idx] + rng.uniform(-30, 30, size=30)

    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    beta_lts, obj = fast_lts(X, y, alpha=0.75, n_starts=200)
    print(f"  True beta      : {beta_true}")
    print(f"  OLS  beta      : {np.round(beta_ols, 3)}   err = {np.linalg.norm(beta_ols - beta_true):.3f}")
    print(f"  LTS  beta      : {np.round(beta_lts, 3)}   err = {np.linalg.norm(beta_lts - beta_true):.3f}")
    print(f"  30% of n = {n} outliers injected")


if __name__ == "__main__":
    demo()

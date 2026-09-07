"""Density ratio estimation (Reference Sec 46.18).

Sugiyama, Suzuki & Kanamori 2012 'Density Ratio Estimation in Machine
Learning'. Directly estimate r(x) = p_num(x) / p_den(x) WITHOUT
estimating either density separately -- much easier than the
two-density-then-divide approach.

Methods:
    * KLIEP  -- Kullback-Leibler Importance Estimation (maximise
                 sum log r(x_num_i) subject to E_den[r] = 1)
    * uLSIF  -- unconstrained Least-Squares Importance Fitting
                 (min ||r - r_true||^2 quadratic form)
    * LR/NCE -- fit classifier (num vs den); r_hat(x) = p1/(1-p1) * n_den/n_num

Applications:
    * Covariate-shift adaptation (importance weights).
    * Change-point / drift detection.
    * Two-sample tests.
    * Off-policy evaluation (importance sampling).
    * Mutual-information / KL estimation.

We implement uLSIF for a Gaussian basis and the classifier-based
estimator, then compare against the analytic ratio.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def gaussian_basis(x, centers, sigma):
    """phi_k(x) = exp(-||x - c_k||^2 / (2 sigma^2)).  x, centers 1-D scalars."""
    x = np.atleast_1d(x).reshape(-1, 1)
    c = np.atleast_1d(centers).reshape(1, -1)
    return np.exp(-((x - c) ** 2) / (2 * sigma ** 2))


def ulsif(x_num, x_den, sigma=1.0, lam=1e-3, n_basis=50, seed=0):
    """Unconstrained LSIF: r(x) = sum alpha_k * phi_k(x)."""
    rng = np.random.default_rng(seed)
    centers = x_num[rng.choice(len(x_num), size=min(n_basis, len(x_num)), replace=False)]
    Phi_num = gaussian_basis(x_num, centers, sigma)
    Phi_den = gaussian_basis(x_den, centers, sigma)
    H = Phi_den.T @ Phi_den / len(x_den)
    h = Phi_num.mean(axis=0)
    alpha = np.linalg.solve(H + lam * np.eye(len(centers)), h)
    def r_hat(x):
        return gaussian_basis(x, centers, sigma) @ alpha
    return r_hat


def classifier_ratio(x_num, x_den):
    """Train logistic-regression classifier on num vs den; ratio = odds * n_den/n_num."""
    X = np.r_[x_num.reshape(-1, 1), x_den.reshape(-1, 1)]
    y = np.r_[np.ones(len(x_num)), np.zeros(len(x_den))]
    #  fit logistic by IRLS or simple gradient
    Xd = np.c_[np.ones(len(X)), X]
    beta = np.zeros(Xd.shape[1])
    for _ in range(200):
        p = 1 / (1 + np.exp(-Xd @ beta))
        grad = Xd.T @ (y - p)
        H = Xd.T @ np.diag(p * (1 - p)) @ Xd
        beta += np.linalg.solve(H + 1e-3 * np.eye(Xd.shape[1]), grad)
    def r_hat(x):
        p1 = 1 / (1 + np.exp(-np.c_[np.ones(len(x)), x] @ beta))
        return (p1 / (1 - p1)) * (len(x_den) / len(x_num))
    return r_hat


if __name__ == "__main__":
    print("=== Density ratio estimation (uLSIF + classifier) ===\n")
    rng = np.random.default_rng(0)
    #  p_num = N(0, 1); p_den = N(1, 1.5)
    x_num = rng.normal(0, 1, size=1000)
    x_den = rng.normal(1, 1.5, size=1000)

    r_ulsif = ulsif(x_num, x_den, sigma=0.6, lam=1e-3, n_basis=50)
    r_clf = classifier_ratio(x_num, x_den)

    #  Analytic ratio at test points
    from scipy.stats import norm
    x_test = np.linspace(-3, 4, 8)
    r_true = norm.pdf(x_test, 0, 1) / norm.pdf(x_test, 1, 1.5)

    print(f"  {'x':>6s}  {'true r':>10s}  {'uLSIF':>10s}  {'classifier':>12s}")
    for xt, rt in zip(x_test, r_true):
        ru = float(r_ulsif(np.array([xt]))[0])
        rc = float(r_clf(np.array([xt]))[0])
        print(f"  {xt:>6.2f}  {rt:>10.3f}  {ru:>10.3f}  {rc:>12.3f}")

    print("\n--- library cross-check (densratio R; densratio Python) ---")

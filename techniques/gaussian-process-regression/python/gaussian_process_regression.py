"""Gaussian Process regression (Reference §14.32; Rasmussen-Williams 2006).

Nonparametric Bayesian regression: place a Gaussian process prior over
the unknown function f, condition on data.

    f ~ GP(m(x), k(x, x'))
    y_i = f(x_i) + eps_i,  eps_i ~ N(0, sigma_n^2)

Given training (X, y), the posterior over f at test points X_* is Gaussian:
    mean:  K(X_*, X) [K(X, X) + sigma_n^2 I]^-1 y
    var :  K(X_*, X_*) - K(X_*, X) [K(X, X) + sigma_n^2 I]^-1 K(X, X_*)

RBF (squared-exponential) kernel:
    k(x, x') = sigma_f^2 exp(-||x - x'||^2 / (2 l^2))
Hyperparameters (l, sigma_f, sigma_n) tuned by MAXIMIZING LOG MARGINAL LIKELIHOOD:
    log p(y | X) = -0.5 y^T K_y^-1 y - 0.5 log |K_y| - (n/2) log(2 pi)

Cost O(n^3) for the Cholesky.  Sparse GPs, inducing points, and kernel-
approximation methods scale to larger n.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy.optimize import minimize    # SciPy optimizer (BFGS/Newton) for MLE


def rbf_kernel(X, Y, length_scale, sigma_f):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X ** 2, 1)[:, None] + np.sum(Y ** 2, 1)[None, :] - 2 * X @ Y.T
    return sigma_f ** 2 * np.exp(-0.5 * d2 / length_scale ** 2)


def gp_neg_log_marginal(X, y, length_scale, sigma_f, sigma_n):
    n = len(y)
    K = rbf_kernel(X, X, length_scale, sigma_f) + (sigma_n ** 2) * np.eye(n)
    try:
        L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        return 1e10
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    logdet = 2 * np.sum(np.log(np.diag(L)))
    return 0.5 * float(y @ alpha) + 0.5 * logdet + 0.5 * n * math.log(2 * math.pi)


def gp_fit(X, y, init: dict = None) -> dict:
    """Fit an RBF GP by maximizing marginal likelihood over (length_scale, sigma_f, sigma_n)."""
    X = np.atleast_2d(np.asarray(X, dtype=float)); y = np.asarray(y, dtype=float)
    init = init or {"length_scale": 1.0, "sigma_f": 1.0, "sigma_n": 0.1}
    def neg_ml(log_params):
        ls, sf, sn = np.exp(log_params)
        return gp_neg_log_marginal(X, y, ls, sf, sn)
    log_init = np.log([init["length_scale"], init["sigma_f"], init["sigma_n"]])
    res = minimize(neg_ml, log_init, method="L-BFGS-B")
    ls, sf, sn = np.exp(res.x)
    return {"length_scale": float(ls), "sigma_f": float(sf), "sigma_n": float(sn),
            "log_marginal_likelihood": float(-res.fun)}


def gp_predict(X, y, X_star, hyper: dict) -> dict:
    X = np.atleast_2d(np.asarray(X, dtype=float)); y = np.asarray(y, dtype=float)
    X_star = np.atleast_2d(np.asarray(X_star, dtype=float))
    ls, sf, sn = hyper["length_scale"], hyper["sigma_f"], hyper["sigma_n"]
    K = rbf_kernel(X, X, ls, sf) + (sn ** 2) * np.eye(len(y)) + 1e-8 * np.eye(len(y))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    K_s = rbf_kernel(X, X_star, ls, sf)
    mu = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    var = rbf_kernel(X_star, X_star, ls, sf).diagonal() - np.sum(v ** 2, 0)
    return {"mean": mu, "variance": np.maximum(var, 1e-10)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 20
    X = rng.uniform(-3, 3, n).reshape(-1, 1)
    y = np.sin(X[:, 0] * 1.5) + rng.normal(0, 0.15, n)

    hyper = gp_fit(X, y)
    print(f"=== GP regression fitted hyperparameters ===")
    print(f"  length_scale = {hyper['length_scale']:.3f}")
    print(f"  sigma_f      = {hyper['sigma_f']:.3f}")
    print(f"  sigma_n      = {hyper['sigma_n']:.3f}   (true noise 0.15)")
    print(f"  log ML       = {hyper['log_marginal_likelihood']:.3f}")

    X_star = np.linspace(-4, 4, 5).reshape(-1, 1)
    p = gp_predict(X, y, X_star, hyper)
    print("\n=== Predictions at 5 test points ===")
    for xs, m, v in zip(X_star.ravel(), p["mean"], p["variance"]):
        print(f"  x = {xs:5.2f}  pred = {m:6.3f} +/- {math.sqrt(v):.3f}  (true sin(1.5x) = {math.sin(1.5 * xs):6.3f})")

    print("\n--- library cross-check (sklearn GaussianProcessRegressor) ---")
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(noise_level=0.01)).fit(X, y)
        y_sk, sd_sk = gp.predict(X_star, return_std=True)
        print(f"  sklearn preds: {y_sk.round(3)}")
    except Exception as ex:
        print(f"  (sklearn GPR unavailable: {ex})")

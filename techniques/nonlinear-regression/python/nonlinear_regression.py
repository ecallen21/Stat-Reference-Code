"""Nonlinear least squares regression (Reference §5.13).

Model:
    y_i = f(x_i, theta) + eps_i,   eps_i ~ N(0, sigma^2)

Minimize sum of squared residuals over the parameter vector theta:
    theta_hat = argmin sum_i (y_i - f(x_i, theta))^2

Contrast with linear regression: f is a KNOWN parametric form (Michaelis-
Menten, sigmoid, exponential decay, Hill, ...); parameters usually have
substantive units.

Solver: Levenberg-Marquardt (Marquardt 1963) -- damped Gauss-Newton that
interpolates between gradient descent (far from optimum) and Gauss-Newton
(close to optimum).

    Gauss-Newton step:  (J^T J) delta = J^T r
    Levenberg-Marquardt: (J^T J + lambda I) delta = J^T r
        lambda large  -> gradient-descent behavior
        lambda small  -> Gauss-Newton behavior

Asymptotic covariance of theta_hat under Normal errors:
    Cov(theta_hat) = sigma_hat^2 * (J^T J)^-1

where J is the Jacobian d f / d theta at theta_hat and sigma_hat^2 = RSS / (n - p).

Requires good starting values -- multiple local minima are common.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def _num_jacobian(f, theta, x, eps: float = 1e-6):
    """Central-difference Jacobian d f / d theta at parameter theta."""
    p = len(theta); n = len(x)
    J = np.zeros((n, p))
    for k in range(p):
        e = np.zeros(p); e[k] = eps
        J[:, k] = (f(x, theta + e) - f(x, theta - e)) / (2 * eps)
    return J


def nlsq_lm(f, x, y, theta0, max_iter: int = 100, tol: float = 1e-8,
            lam0: float = 1e-3) -> dict:
    """Levenberg-Marquardt nonlinear least squares (numerical Jacobian).

    f      : callable f(x, theta) -> vector of predictions.
    theta0 : starting parameter vector.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    theta = np.array(theta0, dtype=float); n = len(y); p = len(theta)
    lam = float(lam0)
    r = y - f(x, theta); ssq = float(r @ r)
    for it in range(max_iter):
        J = _num_jacobian(f, theta, x)
        A = J.T @ J + lam * np.eye(p)
        g = J.T @ r
        try:
            delta = np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            lam *= 10; continue
        theta_new = theta + delta
        r_new = y - f(x, theta_new); ssq_new = float(r_new @ r_new)
        if ssq_new < ssq:
            theta, r, ssq = theta_new, r_new, ssq_new
            lam = max(lam / 3, 1e-12)
        else:
            lam *= 3
        if np.max(np.abs(delta)) < tol * (1 + np.linalg.norm(theta)): break
    # Standard errors
    J = _num_jacobian(f, theta, x)
    sigma2 = ssq / (n - p) if n > p else float("nan")
    cov = sigma2 * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.diag(cov))
    return {"theta": theta, "se": se,
            "residual_ss": ssq, "sigma2": float(sigma2),
            "df_resid": int(n - p),
            "z": theta / se, "p_value": 2 * stats.norm.sf(np.abs(theta / se)),
            "iterations": int(it + 1), "lambda_final": float(lam),
            "method": "Nonlinear least squares (Levenberg-Marquardt, numerical Jacobian)"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Michaelis-Menten:  y = Vmax * x / (Km + x) + noise
    def mm(x, theta):
        Vmax, Km = theta
        return Vmax * x / (Km + x)

    Vmax_true, Km_true = 5.0, 2.0
    x = np.linspace(0.1, 20, 40)
    y = mm(x, [Vmax_true, Km_true]) + rng.normal(0, 0.2, len(x))

    print("=== Michaelis-Menten nonlinear least squares ===")
    r = nlsq_lm(mm, x, y, theta0=[1.0, 1.0])
    for name, est, se, p in zip(["Vmax", "Km"], r["theta"], r["se"], r["p_value"]):
        print(f"  {name}: {est:.3f} (SE {se:.3f}, p = {p:.3g})   true = "
              f"{Vmax_true if name == 'Vmax' else Km_true}")
    print(f"  RSS = {r['residual_ss']:.4f}, sigma^2 = {r['sigma2']:.4f}")
    print(f"  iterations = {r['iterations']}")

    # Sigmoid dose-response:  y = bottom + (top - bottom) / (1 + exp(-slope (x - EC50)))
    print("\n=== Sigmoid four-parameter dose-response ===")
    def sigm(x, theta):
        bottom, top, slope, EC50 = theta
        return bottom + (top - bottom) / (1 + np.exp(-slope * (x - EC50)))
    theta_t = [0.1, 1.0, 1.5, 3.0]
    x = np.linspace(0, 8, 60)
    y = sigm(x, theta_t) + rng.normal(0, 0.03, len(x))
    r = nlsq_lm(sigm, x, y, theta0=[0.0, 1.0, 1.0, 2.0])
    for name, est, se in zip(["bottom", "top", "slope", "EC50"], r["theta"], r["se"]):
        print(f"  {name:8s}: {est:.3f} (SE {se:.3f})   true = {theta_t[['bottom','top','slope','EC50'].index(name)]}")

    print("\n--- library cross-check (scipy.optimize.curve_fit) ---")
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(lambda x, V, K: V * x / (K + x),
                                np.linspace(0.1, 20, 40),
                                mm(np.linspace(0.1, 20, 40), [Vmax_true, Km_true]) +
                                np.random.default_rng(0).normal(0, 0.2, 40),
                                p0=[1, 1])
        print(f"  scipy Vmax = {popt[0]:.3f}, Km = {popt[1]:.3f}")
    except Exception as ex:
        print(f"  (scipy curve_fit unavailable: {ex})")

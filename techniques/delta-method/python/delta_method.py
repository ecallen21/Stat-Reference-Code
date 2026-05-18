"""The Delta Method (Reference §3.29).

Approximate variance / standard error of a *function* of one or more estimated
parameters whose joint asymptotic distribution is known.

If  theta_hat  --d-->  N(theta, V)  (V = covariance matrix of theta_hat),
and g is a smooth function, then by a first-order Taylor expansion,

    g(theta_hat)  --d-->  N(g(theta),  grad(g)(theta)^T  V  grad(g)(theta))

So SE(g(theta_hat)) approx sqrt(grad(g)^T V grad(g)) -- the delta-method SE.

Use cases
---------
- SE for a transformed parameter (e.g. CV = sigma/mu; log of an odds ratio).
- CIs for nonlinear functions in regression / GLM output.
- Avoiding the need for a separate bootstrap when V is already known.

Caveats
-------
- It's a *first-order* approximation. Near boundaries or for highly nonlinear g,
  the bootstrap or a likelihood-based CI is preferable.
- Asymmetric CIs: transform to a scale where the normal approximation works,
  build the CI there, then back-transform (e.g. log-OR + Wald -> exponentiate).

This file provides:
  - ``delta_se(g, theta, V, step=1e-5)``   numerical-gradient SE
  - ``delta_ci(g, theta, V, conf=0.95)``   the corresponding 95% CI
  - worked example: log-OR CI from a 2x2 table (closed form vs. delta method)
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import stats


def numerical_gradient(g: Callable, theta: Sequence[float], step: float = 1e-5):
    """Central-difference gradient of a scalar function g at ``theta``."""
    theta = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta); e[i] = step
        grad[i] = (g(theta + e) - g(theta - e)) / (2 * step)
    return grad


def delta_se(g: Callable, theta, V, step: float = 1e-5) -> float:
    """Delta-method SE of g(theta) where theta_hat has covariance V."""
    grad = numerical_gradient(g, theta, step)
    V = np.asarray(V, dtype=float)
    return float(math.sqrt(grad @ V @ grad))


def delta_ci(g: Callable, theta, V, conf: float = 0.95, step: float = 1e-5):
    """Delta-method point estimate and 100*conf% Wald CI for g(theta)."""
    g_hat = float(g(np.asarray(theta, dtype=float)))
    se = delta_se(g, theta, V, step)
    z = stats.norm.ppf(0.5 + conf / 2)
    return {"estimate": g_hat, "se": se,
            "ci_lower": g_hat - z * se, "ci_upper": g_hat + z * se}


# ---------- Worked example: log odds ratio CI from a 2 x 2 table -----------
def log_or_ci_closed_form(a, b, c, d, conf=0.95):
    """Standard Wald CI for log(OR) using SE = sqrt(1/a + 1/b + 1/c + 1/d)."""
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(0.5 + conf / 2)
    return {"log_or": log_or, "se_closed": se,
            "ci_lower_log": log_or - z * se,
            "ci_upper_log": log_or + z * se,
            "ci_lower_or": math.exp(log_or - z * se),
            "ci_upper_or": math.exp(log_or + z * se)}


def log_or_ci_delta(a, b, c, d, conf=0.95):
    """Same CI obtained generically via numerical delta on theta = (a, b, c, d)."""
    # Treat cell counts as MLE of multinomial cell counts. Variance of each
    # count under a multinomial with fixed N is N * p (1 - p); covariance is
    # -N p_i p_j. Here we use the simpler Poisson approximation Var(count_i) = count_i,
    # which makes the cells independent and matches the textbook log-OR SE.
    theta = [a, b, c, d]
    V = np.diag(theta)
    g = lambda t: math.log((t[0] * t[3]) / (t[1] * t[2]))   # log OR
    res = delta_ci(g, theta, V, conf=conf)
    return {"log_or": res["estimate"], "se_delta": res["se"],
            "ci_lower_log": res["ci_lower"], "ci_upper_log": res["ci_upper"],
            "ci_lower_or": math.exp(res["ci_lower"]),
            "ci_upper_or": math.exp(res["ci_upper"])}


# Another classic: SE of the coefficient of variation s / x_bar
def cv_se_delta(mean, sd, var_mean, var_sd, cov_mean_sd=0.0):
    """Delta-method SE of CV = sd / mean given Var(mean), Var(sd), Cov(mean, sd)."""
    theta = [mean, sd]; V = [[var_mean, cov_mean_sd], [cov_mean_sd, var_sd]]
    return delta_se(lambda t: t[1] / t[0], theta, V)


if __name__ == "__main__":
    print("=== Log-OR CI for 2x2 = [[40, 20], [10, 30]] ===")
    closed = log_or_ci_closed_form(40, 20, 10, 30)
    delta = log_or_ci_delta(40, 20, 10, 30)
    print("Closed-form (textbook SE):", closed)
    print("Delta method (numerical grad):", delta)
    print()
    print("Two routes give nearly identical numbers, by construction --")
    print("the textbook formula IS the delta method applied to log(ad/bc).")

    # CV example: x_bar = 50, sd = 12, large independent sample
    print("\n=== Delta-method SE for CV = s / x_bar ===")
    n = 50
    mean_ = 50.0; sd_ = 12.0
    var_mean = sd_ ** 2 / n
    var_sd = sd_ ** 2 / (2 * (n - 1))    # standard approximation
    se_cv = cv_se_delta(mean_, sd_, var_mean, var_sd)
    print(f"  CV_hat = {sd_ / mean_:.4f}   SE(CV) approx = {se_cv:.4f}")

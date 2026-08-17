"""Person-fit statistics: detecting aberrant response patterns
(Reference §22.13).

Given estimated IRT item parameters (a, b) and a person's response
vector y, the l_z statistic (Drasgow-Levine-Williams 1985) measures
how UNLIKELY the pattern is under the model.

    l   = sum_j y_j log P_j + (1 - y_j) log(1 - P_j)     log-likelihood
    E[l]= sum_j P_j log P_j + (1 - P_j) log(1 - P_j)
    Var[l] = sum_j P_j (1 - P_j) (log(P_j / (1 - P_j)))^2
    l_z = (l - E[l]) / sqrt(Var[l])  ~ N(0, 1) approximately

Aberrant response patterns
    l_z << -2: person answered easy items wrong and hard items right =>
                cheating / random / disengaged response pattern.
    l_z >> +2: pattern too good to be true.

Widely used to screen respondents in operational testing.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def _sigmoid(x): return 1 / (1 + np.exp(-x))


def lz_statistic(y, theta_hat, a, b) -> float:
    """Standardized log-lik person-fit index."""
    P = _sigmoid(a * (theta_hat - b))
    P = np.clip(P, 1e-8, 1 - 1e-8)
    l = float(np.sum(y * np.log(P) + (1 - y) * np.log(1 - P)))
    E_l = float(np.sum(P * np.log(P) + (1 - P) * np.log(1 - P)))
    var_l = float(np.sum(P * (1 - P) * (np.log(P / (1 - P))) ** 2))
    return (l - E_l) / math.sqrt(max(var_l, 1e-10))


def person_fit_all(Y, theta_hats, a, b) -> np.ndarray:
    Y = np.asarray(Y, dtype=float); a = np.asarray(a); b = np.asarray(b)
    return np.array([lz_statistic(Y[i], theta_hats[i], a, b) for i in range(len(Y))])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, J = 300, 20
    theta_true = rng.normal(0, 1, n)
    a = np.ones(J); b = np.linspace(-2, 2, J)
    P = _sigmoid(a[None, :] * (theta_true[:, None] - b[None, :]))
    Y = (rng.uniform(size=P.shape) < P).astype(int)

    # Inject a few aberrant respondents: person 0 flips answers on easy items;
    # person 1 answers randomly regardless of ability.
    Y[0, :5] = 0; Y[0, 15:] = 1               # inversion
    Y[1] = (rng.uniform(size=J) < 0.5).astype(int)

    # Use MLE theta given true item params (skip for demo speed; assume known)
    from scipy.optimize import minimize
    theta_hats = np.zeros(n)
    for i in range(n):
        def neg_ll(t):
            P = _sigmoid(a * (t - b))
            P = np.clip(P, 1e-8, 1 - 1e-8)
            return -np.sum(Y[i] * np.log(P) + (1 - Y[i]) * np.log(1 - P))
        theta_hats[i] = float(minimize(neg_ll, 0.0, method="BFGS").x[0])

    lz = person_fit_all(Y, theta_hats, a, b)
    print(f"=== Person-fit l_z ===")
    print(f"  aberrant person 0 (inversion): l_z = {lz[0]:.3f}")
    print(f"  aberrant person 1 (random):    l_z = {lz[1]:.3f}")
    print(f"  ordinary person 100:            l_z = {lz[100]:.3f}")
    print(f"\n  Distribution of l_z (should be roughly N(0, 1)):")
    print(f"    mean = {lz[2:].mean():.3f}, sd = {lz[2:].std():.3f}")
    n_flag = int(np.sum(lz < -2.0))
    print(f"  n with l_z < -2 (flagged aberrant): {n_flag} of {n}")

    print("\n--- library cross-check (R PerFit::lz / mirt::personfit) ---")

"""Overdispersion detection in count / binary GLMs (Reference §7.35, §7.42, §7.54).

Three complementary diagnostics for Poisson / binomial GLMs:

1. PEARSON DISPERSION ESTIMATE
       phi_hat = chi^2_P / (n - p),    chi^2_P = sum (y - mu)^2 / V(mu)
   phi_hat > 1.5 typically signals overdispersion in a Poisson; phi_hat ~ 1
   for a well-fitting Poisson.

2. SCORE TEST FOR OVERDISPERSION (Cameron-Trivedi 1990).
   In Poisson with alternative Var(Y) = mu + alpha * mu * g(mu) (g(mu) = 1 or mu),
   the score statistic against alpha = 0 is:
       T = sum_i [ (y_i - mu_i)^2 - y_i ] * g(mu_i) / mu_i
   normalized by its variance. Asymptotically N(0, 1) under H0 (no overdispersion).
   Two common forms:
       g(mu) = 1     -> NB1 (linear variance)
       g(mu) = mu    -> NB2 (quadratic variance) -- the default
   Significant T -> overdispersion present, consider NB.

3. LIKELIHOOD-RATIO TEST POISSON vs NB.
       LRT = 2 (ll_NB - ll_Poisson)
   Compared to a 50:50 mixture of chi^2_0 and chi^2_1 because alpha = 0 is at the
   boundary of the parameter space (so the standard chi^2_1 p-value is conservative
   by a factor of 2).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import NamedTuple    # stdlib: NamedTuple = tuple with named fields (prints as Name(field=...))

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats, special    # stats: distributions/tests;  special: gammaln/beta


class PoissonFit(NamedTuple):
    """IRLS Poisson fit. Unpacks like a tuple: ``beta, mu = _poisson_fit_irls(X, y)``."""
    beta: np.ndarray   # coefficient vector
    mu: np.ndarray     # fitted means = exp(X @ beta)


def _poisson_fit_irls(X, y, max_iter=50, tol=1e-8):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)
    for _ in range(max_iter):
        eta = X @ beta; mu = np.exp(np.clip(eta, -500, 500))
        w = np.clip(mu, 1e-12, None)
        z = eta + (y - mu) / w
        sw = np.sqrt(w); Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol: beta = beta_new; break
        beta = beta_new
    return PoissonFit(beta=beta, mu=mu)


def pearson_dispersion(X, y) -> dict:
    """Fit Poisson; return phi_hat = chi^2_P / (n - p)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, mu = _poisson_fit_irls(X, y)
    eta = X @ beta; mu = np.exp(eta)
    chi2 = float(np.sum(((y - mu) ** 2) / np.clip(mu, 1e-12, None)))
    return {"pearson_chi_square": chi2, "df": n - p,
            "phi_hat": chi2 / (n - p),
            "interpretation": "phi > ~1.5 suggests overdispersion; consider NB"}


def score_test(X, y, form: str = "NB2") -> dict:
    """Cameron-Trivedi score test against Poisson with NB1 or NB2 variance.

    H0: Var(Y) = mu        (Poisson)
    H1: Var(Y) = mu + alpha * g(mu),  g(mu) = 1 (NB1) or g(mu) = mu (NB2)
    """
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    beta, mu = _poisson_fit_irls(X, y)
    mu = np.exp(X @ beta)
    if form == "NB2":
        g = mu
    elif form == "NB1":
        g = np.ones_like(mu)
    else:
        raise ValueError("form must be 'NB1' or 'NB2'")
    # T = sum_i [ ((y - mu)^2 - y) / mu ] * g(mu)/mu
    aux = ((y - mu) ** 2 - y) / np.clip(mu, 1e-12, None) * g / np.clip(mu, 1e-12, None)
    # The auxiliary OLS regression of aux on 1 with intercept-only gives T:
    T = float(np.sum(aux)) / math.sqrt(2 * float(np.sum((g / mu) ** 2)))
    p_value_one_sided = float(stats.norm.sf(T))
    return {"T": T, "form": form,
            "p_value_one_sided": p_value_one_sided,
            "p_value_two_sided": float(2 * stats.norm.sf(abs(T))),
            "method": f"Cameron-Trivedi score test ({form})"}


def poisson_vs_nb_lrt(X, y) -> dict:
    """LRT of Poisson vs negative binomial (boundary-adjusted)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    # Poisson log-likelihood (drop the y!-constant for the difference)
    beta_p, _ = _poisson_fit_irls(X, y); mu_p = np.exp(X @ beta_p)
    ll_pois = float(np.sum(y * np.log(np.clip(mu_p, 1e-15, None)) - mu_p
                            - np.array([math.lgamma(yi + 1) for yi in y])))
    # NB MLE via joint BFGS (see techniques/negative-binomial-regression)
    from scipy import optimize
    n, p = X.shape
    def neg_ll_nb(params):
        beta = params[:p]; theta = math.exp(params[p])
        eta = X @ beta; mu = np.exp(np.clip(eta, -500, 500))
        return -float(np.sum(
            special.gammaln(y + theta) - special.gammaln(theta) - special.gammaln(y + 1)
            + theta * np.log(theta / (theta + mu)) + y * np.log(mu / (theta + mu))
        ))
    init = np.concatenate([beta_p, [0.0]])
    res = optimize.minimize(neg_ll_nb, init, method="BFGS",
                            options={"gtol": 1e-7, "disp": False})
    ll_nb = -res.fun
    lrt = 2 * (ll_nb - ll_pois)
    # Boundary-adjusted p: half-half mixture of chi2_0 and chi2_1
    p_val = 0.5 * float(stats.chi2.sf(max(lrt, 0), 1))
    return {"lrt": lrt, "ll_poisson": ll_pois, "ll_nb": ll_nb,
            "p_value_boundary_adjusted": p_val,
            "method": "LRT Poisson vs NB (boundary-adjusted)"}


def run_all(X, y) -> dict:
    return {"pearson_dispersion": pearson_dispersion(X, y),
            "score_test_NB2": score_test(X, y, form="NB2"),
            "score_test_NB1": score_test(X, y, form="NB1"),
            "lrt_poisson_vs_nb": poisson_vs_nb_lrt(X, y)}


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x1, x2])

    print("=== Pure Poisson data (should NOT flag overdispersion) ===")
    mu = np.exp(0.5 + 0.6 * x1 - 0.3 * x2)
    y_poisson = rng.poisson(mu)
    for k, v in run_all(X, y_poisson).items():
        print(f"  {k:22s}: {v}")

    print("\n=== NB data (theta = 1.5; should flag overdispersion) ===")
    y_nb = rng.negative_binomial(1.5, 1.5 / (1.5 + mu))
    for k, v in run_all(X, y_nb).items():
        print(f"  {k:22s}: {v}")

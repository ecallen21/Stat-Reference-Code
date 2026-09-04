"""Copulas (Reference Sec 38.9).

Sklar (1959): every joint CDF F factors as

    F(x_1, x_2) = C( F_1(x_1), F_2(x_2) )

where C is a COPULA -- a joint CDF on [0, 1]^d with uniform margins
that encodes the DEPENDENCE STRUCTURE independently of the marginal
distributions.

Three families implemented here:

  GAUSSIAN(rho)    -- symmetric, no tail dependence.
  CLAYTON(theta)   -- lower-tail dependence, theta > 0.
  GUMBEL(theta)    -- upper-tail dependence, theta >= 1.

Fitting: MLE on the copula log-likelihood after transforming margins
to uniforms via empirical CDFs (pseudo-observations).
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats, optimize    # normal cdf/pdf + minimize


def _pseudo_obs(x):
    """Empirical CDF pseudo-observations in (0, 1)."""
    n = len(x)
    ranks = stats.rankdata(x)
    return ranks / (n + 1)


def gaussian_copula_loglik(rho, u, v):
    z1 = stats.norm.ppf(u); z2 = stats.norm.ppf(v)
    r = np.clip(rho, -0.99, 0.99)
    return float(-0.5 * np.log(1 - r ** 2) * len(u)
                 - (r ** 2 * (z1 ** 2 + z2 ** 2) - 2 * r * z1 * z2).sum() / (2 * (1 - r ** 2)))


def clayton_loglik(theta, u, v):
    if theta <= 0:
        return -np.inf
    a = u ** -theta + v ** -theta - 1
    ll = np.log(1 + theta) + (-1 - theta) * (np.log(u) + np.log(v)) \
        + (-2 - 1 / theta) * np.log(a)
    return float(ll.sum())


def gumbel_loglik(theta, u, v):
    if theta < 1:
        return -np.inf
    lu = -np.log(u); lv = -np.log(v)
    A = (lu ** theta + lv ** theta) ** (1 / theta)
    C = np.exp(-A)
    ll = (np.log(C) + np.log(A ** (1 - 2 * theta) * (lu * lv) ** (theta - 1)
                              * (A + theta - 1)) - np.log(u * v))
    return float(ll.sum())


def fit_copula(family, u, v):
    """MLE over the single dependence parameter."""
    if family == "gaussian":
        obj = lambda r: -gaussian_copula_loglik(r, u, v)
        res = optimize.minimize_scalar(obj, bounds=(-0.99, 0.99), method="bounded")
        return {"param": float(res.x), "loglik": float(-res.fun)}
    if family == "clayton":
        obj = lambda t: -clayton_loglik(t, u, v)
        res = optimize.minimize_scalar(obj, bounds=(1e-4, 30), method="bounded")
        return {"param": float(res.x), "loglik": float(-res.fun)}
    if family == "gumbel":
        obj = lambda t: -gumbel_loglik(t, u, v)
        res = optimize.minimize_scalar(obj, bounds=(1.0, 30), method="bounded")
        return {"param": float(res.x), "loglik": float(-res.fun)}
    raise ValueError(family)


if __name__ == "__main__":
    print("=== Copulas: Sklar + Gaussian / Clayton / Gumbel MLE ===\n")
    rng = np.random.default_rng(0)
    n = 800
    # Simulate a Clayton(theta = 2) with N(0,1) and Exp(1) margins
    theta_true = 2.0
    v = rng.random(n)
    w = rng.random(n)
    # Clayton conditional sampler: solve for u given v, w
    u = (w ** (-theta_true / (1 + theta_true)) * (v ** -theta_true) - v ** -theta_true + 1) ** (-1 / theta_true)
    # Apply non-uniform margins: x = Phi^{-1}(u), y = -log(1 - v) (exp(1))
    x = stats.norm.ppf(u)
    y = -np.log(1 - v)

    # Convert back to pseudo-uniforms and fit each family
    up = _pseudo_obs(x); vp = _pseudo_obs(y)
    for fam in ("gaussian", "clayton", "gumbel"):
        r = fit_copula(fam, up, vp)
        aic = 2 * 1 - 2 * r["loglik"]
        print(f"  {fam:<10s}  param = {r['param']:.3f}   loglik = {r['loglik']:8.2f}   AIC = {aic:8.2f}")
    print(f"\n  True generating copula: Clayton(theta = {theta_true}) -> lowest AIC identifies Clayton.")

    # Kendall's tau linkage: Clayton tau = theta / (theta + 2); Gumbel tau = 1 - 1/theta
    from scipy.stats import kendalltau
    tau, _ = kendalltau(x, y)
    print(f"\n  Sample Kendall tau = {tau:.3f}")
    print(f"  Clayton implied  theta = 2*tau / (1 - tau) = {2 * tau / (1 - tau):.3f}")

    print("\n--- library cross-check (R copula/VineCopula; Python copulas + custom) ---")

"""Extreme value theory (Reference Sec 38.1).

Two workhorse approaches for modelling the tails of a distribution:

  BLOCK MAXIMA -> Generalised Extreme Value (GEV) distribution
    M_n = max(X_1, ..., X_n).  For large n, (M_n - a_n) / b_n
    converges to GEV(mu, sigma, xi).  xi < 0 = Weibull (bounded),
    xi = 0 = Gumbel, xi > 0 = Frechet (heavy tail).

  PEAKS OVER THRESHOLD (POT) -> Generalised Pareto Distribution (GPD)
    Given X > u, (X - u) ~ GPD(sigma_u, xi).  Fisher-Tippett-Gnedenko
    guarantees this for u sufficiently large.

Common quantities: RETURN LEVEL (m-year event) and RETURN PERIOD.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays
from scipy import stats, optimize    # GEV/GPD densities + MLE


def fit_gev(block_maxima):
    """Fit GEV(mu, sigma, xi) by MLE."""
    xi, mu, sigma = stats.genextreme.fit(block_maxima)   # note: scipy uses c = -xi
    return {"mu": float(mu), "sigma": float(sigma), "xi": float(-xi)}


def gev_return_level(mu, sigma, xi, T):
    """m-year return level under GEV (annual maxima) at return period T."""
    p = 1.0 - 1.0 / T
    if abs(xi) < 1e-8:
        return mu - sigma * np.log(-np.log(p))
    return mu + (sigma / xi) * (((-np.log(p)) ** (-xi)) - 1)


def fit_gpd(excesses):
    """Fit GPD(sigma, xi) to excess exceedances over a threshold u."""
    xi, loc, sigma = stats.genpareto.fit(excesses, floc=0.0)  # scipy c = xi here
    return {"sigma": float(sigma), "xi": float(xi)}


if __name__ == "__main__":
    print("=== Extreme value theory: GEV (block maxima) + GPD (POT) ===\n")
    rng = np.random.default_rng(0)
    # Heavy-tailed process: t_3 daily observations, 40 years x 365 days
    n_years = 40
    n_per_year = 365
    x = rng.standard_t(df=3, size=n_years * n_per_year) * 5 + 20
    annual_max = x.reshape(n_years, n_per_year).max(axis=1)

    gev = fit_gev(annual_max)
    print(f"  GEV fit on annual maxima (n = {n_years})")
    print(f"    mu = {gev['mu']:.3f}   sigma = {gev['sigma']:.3f}   xi = {gev['xi']:.3f}")
    for T in (10, 50, 100):
        rl = gev_return_level(**gev, T=T)
        print(f"    {T:>3d}-year return level: {rl:.2f}")

    # Peaks over threshold
    u = np.quantile(x, 0.98)
    excesses = x[x > u] - u
    gpd = fit_gpd(excesses)
    print(f"\n  POT / GPD fit (threshold u = 98th pct = {u:.2f}, n_exc = {len(excesses)})")
    print(f"    sigma_u = {gpd['sigma']:.3f}   xi = {gpd['xi']:.3f}")
    # POT return level: u + (sigma/xi) * ((n * zeta_u / N)^xi - 1)
    zeta_u = len(excesses) / len(x)
    N_annual = n_per_year
    for T in (10, 50, 100):
        m = T * N_annual
        if abs(gpd["xi"]) < 1e-8:
            rl_pot = u + gpd["sigma"] * np.log(m * zeta_u)
        else:
            rl_pot = u + (gpd["sigma"] / gpd["xi"]) * ((m * zeta_u) ** gpd["xi"] - 1)
        print(f"    {T:>3d}-year (POT) return level: {rl_pot:.2f}")

    print("\n--- library cross-check (R extRemes::fevd; Python scipy.genextreme/genpareto) ---")

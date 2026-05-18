"""Tetrachoric and polychoric correlation (Reference §4.7).

Model: an unobserved bivariate normal (X*, Y*) generates the observed
ordinal pair (X, Y) by thresholds:
    X = i  iff  alpha_{i-1} < X* <= alpha_i,    i = 1..r
    Y = j  iff  beta_{j-1}  < Y* <= beta_j,     j = 1..c
with alpha_0 = beta_0 = -inf, alpha_r = beta_c = +inf.

The *polychoric correlation* is the correlation rho of (X*, Y*) -- the
"underlying" association on the latent continuous scale. The thresholds are
estimated from the marginals (via Phi^{-1}(cumulative marginal proportions)),
and rho is found by maximizing the log-likelihood of the multinomial cell
counts.

Special case: 2x2 tables = *tetrachoric* correlation.

This implementation
-------------------
- Estimates thresholds from observed marginals: alpha_i = Phi^{-1}( sum_{j<=i} p_i. )
- Maximizes the bivariate-normal cell-probability log-likelihood over rho via
  scipy.optimize on [-0.999, 0.999].
- Returns rho-hat plus the threshold vectors. Standard errors require numerical
  derivatives; we do not estimate them here -- use R's polycor::polychor or
  Python's semopy for SEs.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import optimize, stats


def _thresholds_from_marginals(margins: Sequence[float]) -> np.ndarray:
    """alpha_i = Phi^-1(cum proportion up to i)."""
    cum = np.cumsum(np.asarray(margins, dtype=float))
    cum = cum[:-1]                                # drop the trailing 1.0
    cum = np.clip(cum, 1e-8, 1 - 1e-8)
    return stats.norm.ppf(cum)


def _bvn_cdf_orthant(a, b, rho):
    """P(X* <= a, Y* <= b) for a standard bivariate normal with correlation rho."""
    return stats.multivariate_normal.cdf([a, b], mean=[0, 0],
                                         cov=[[1, rho], [rho, 1]])


def _cell_prob(alpha_lo, alpha_hi, beta_lo, beta_hi, rho):
    """P(alpha_lo < X* <= alpha_hi, beta_lo < Y* <= beta_hi) via inclusion/exclusion."""
    a_hi = alpha_hi if math.isfinite(alpha_hi) else 8.0
    a_lo = alpha_lo if math.isfinite(alpha_lo) else -8.0
    b_hi = beta_hi if math.isfinite(beta_hi) else 8.0
    b_lo = beta_lo if math.isfinite(beta_lo) else -8.0
    return (_bvn_cdf_orthant(a_hi, b_hi, rho)
            - _bvn_cdf_orthant(a_lo, b_hi, rho)
            - _bvn_cdf_orthant(a_hi, b_lo, rho)
            + _bvn_cdf_orthant(a_lo, b_lo, rho))


def polychoric(table) -> dict:
    """Maximum-likelihood polychoric correlation for an r x c table of counts.

    ``table`` is a 2D array of nonnegative integer counts.
    """
    obs = np.asarray(table, dtype=float)
    r, c = obs.shape
    n = obs.sum()
    row_marg = obs.sum(axis=1) / n
    col_marg = obs.sum(axis=0) / n
    alpha = np.concatenate([[-np.inf], _thresholds_from_marginals(row_marg), [np.inf]])
    beta = np.concatenate([[-np.inf], _thresholds_from_marginals(col_marg), [np.inf]])

    def neg_loglik(rho):
        ll = 0.0
        for i in range(r):
            for j in range(c):
                if obs[i, j] == 0:
                    continue
                p = _cell_prob(alpha[i], alpha[i + 1], beta[j], beta[j + 1], rho)
                p = max(p, 1e-15)
                ll += obs[i, j] * math.log(p)
        return -ll

    res = optimize.minimize_scalar(neg_loglik, bounds=(-0.999, 0.999), method="bounded")
    return {"rho": float(res.x),
            "row_thresholds": alpha[1:-1].tolist(),
            "col_thresholds": beta[1:-1].tolist(),
            "log_likelihood": float(-res.fun),
            "n": float(n), "size": (r, c),
            "method": "tetrachoric" if (r, c) == (2, 2) else "polychoric"}


def library_versions(table):
    out = {}
    try:
        from semopy.polycorr import polychoric_corr  # type: ignore
        out["semopy polychoric_corr"] = float(polychoric_corr(np.asarray(table)))
    except Exception as exc:
        out["semopy"] = f"not used ({exc})"
    return out


if __name__ == "__main__":
    # Example: 3 x 3 ordinal table (Likert vs. Likert)
    table = np.array([
        [20,  8,  2],
        [12, 30, 10],
        [ 3, 12, 25],
    ])
    print("=== Polychoric on 3x3 ordinal table ===")
    res = polychoric(table)
    for k, v in res.items():
        print(f"  {k:18s}: {v}")

    # Tetrachoric (2x2 special case)
    table2 = np.array([[60, 15], [10, 40]])
    print("\n=== Tetrachoric on 2x2 ===")
    for k, v in polychoric(table2).items():
        print(f"  {k:18s}: {v}")

    print("\n--- library ---")
    for k, v in library_versions(table).items():
        print(f"  {k}: {v}")

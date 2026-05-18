"""Cramer's V and the phi coefficient (Reference §4.10).

Effect-size measures of association in a contingency table.

For a 2x2 table:
    phi = sqrt(chi^2 / n)      = (a d - b c) / sqrt((a+b)(c+d)(a+c)(b+d))
    phi in [-1, 1] (signed), and  phi^2 = chi^2 / n.

For an r x c table:
    Cramer's V = sqrt(chi^2 / (n * min(r-1, c-1)))   in [0, 1]
    (V equals |phi| when r = c = 2.)

Bias-corrected V (Bergsma 2013):
    V_tilde = sqrt(max(0, (phi^2 - (r-1)(c-1)/(n-1))) /
                       min(r_tilde - 1, c_tilde - 1))
    with r_tilde = r - (r-1)^2/(n-1), c_tilde = c - (c-1)^2/(n-1).
This corrects the well-known upward bias of V in small samples.

Conventional benchmarks (small/medium/large), df = min(r-1, c-1):
    df = 1: 0.10 / 0.30 / 0.50
    df = 2: 0.07 / 0.21 / 0.35
    df = 3: 0.06 / 0.17 / 0.29
"""
from __future__ import annotations

import math

import numpy as np


def chi_square(table) -> float:
    """Pearson chi-square statistic for an r x c table of counts."""
    obs = np.asarray(table, dtype=float)
    row_tot = obs.sum(axis=1, keepdims=True)
    col_tot = obs.sum(axis=0, keepdims=True)
    n = obs.sum()
    expected = row_tot @ col_tot / n
    mask = expected > 0
    return float(((obs - expected) ** 2 / np.where(mask, expected, 1))[mask].sum())


def phi_coefficient(table) -> float:
    """Phi for a 2x2 table (signed). Returns ``nan`` for non-2x2 tables."""
    obs = np.asarray(table, dtype=float)
    if obs.shape != (2, 2):
        return float("nan")
    a, b, c, d = obs[0, 0], obs[0, 1], obs[1, 0], obs[1, 1]
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return (a * d - b * c) / denom if denom > 0 else float("nan")


def cramers_v(table, bias_correct: bool = False) -> dict:
    """Cramer's V for an r x c table.

    If ``bias_correct``, returns the Bergsma (2013) bias-corrected version too.
    """
    obs = np.asarray(table, dtype=float)
    r, c = obs.shape
    n = obs.sum()
    chi2 = chi_square(obs)
    phi2 = chi2 / n
    V = math.sqrt(phi2 / min(r - 1, c - 1))
    out = {"chi_square": chi2, "n": float(n), "shape": (r, c),
           "phi_squared": phi2, "cramers_v": V}
    if bias_correct:
        r_tilde = r - (r - 1) ** 2 / (n - 1)
        c_tilde = c - (c - 1) ** 2 / (n - 1)
        phi2_corr = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
        denom = min(r_tilde - 1, c_tilde - 1)
        V_corr = math.sqrt(phi2_corr / denom) if denom > 0 else 0.0
        out["cramers_v_bias_corrected"] = V_corr
    if r == 2 and c == 2:
        out["phi_signed"] = phi_coefficient(obs)
    return out


def library_versions(table):
    from scipy.stats import contingency
    return {"scipy.stats.contingency.association (Cramer's V)":
            float(contingency.association(table, method="cramer")),
            "scipy.stats.contingency.association (phi)":
            float(contingency.association(table, method="cramer", correction=False))
            if np.shape(table) == (2, 2) else "(only meaningful for 2x2)"}


if __name__ == "__main__":
    # 2x2 example -> phi and V coincide
    table_2x2 = [[40, 10], [20, 30]]
    print("=== 2x2 table ===")
    for k, v in cramers_v(table_2x2, bias_correct=True).items():
        print(f"  {k:30s}: {v}")

    # 3x4 example
    table_3x4 = [[30, 20, 10,  5],
                 [10, 25, 30, 15],
                 [ 5, 10, 25, 30]]
    print("\n=== 3x4 table ===")
    for k, v in cramers_v(table_3x4, bias_correct=True).items():
        print(f"  {k:30s}: {v}")

    print("\n--- library (scipy.stats.contingency.association) ---")
    for k, v in library_versions(table_3x4).items():
        print(f"  {k}: {v}")

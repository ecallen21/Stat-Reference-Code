"""Generalizability theory (Reference §22.10; Cronbach-Gleser-Nanda-Rajaratnam 1972).

G-theory extends classical reliability by decomposing observed-score variance
into multiple FACETS (raters, items, occasions, ...).

Two-facet crossed design (persons x items x raters), all crossed:
    sigma^2_p     (person)
    sigma^2_i     (item)
    sigma^2_r     (rater)
    sigma^2_{pi}, sigma^2_{pr}, sigma^2_{ir}    (interactions)
    sigma^2_{pir,e}   (residual)

G-study estimates all variance components via random-effects ANOVA.
D-study forecasts reliability for a specific measurement design:
    G coefficient (relative decisions):
        rho^2 = sigma^2_p / (sigma^2_p + sum_error_relative)
    Phi coefficient (absolute decisions):
        Phi = sigma^2_p / (sigma^2_p + sum_error_absolute)

The demo implements the simpler p x i design (one facet: items).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def g_theory_pi(X) -> dict:
    """One-facet crossed p x i design (persons x items).

    X : n_persons x n_items matrix of scores.
    """
    X = np.asarray(X, dtype=float); n_p, n_i = X.shape
    grand = X.mean()
    p_bar = X.mean(axis=1)                                # person means
    i_bar = X.mean(axis=0)                                # item means
    SS_p = n_i * np.sum((p_bar - grand) ** 2)
    SS_i = n_p * np.sum((i_bar - grand) ** 2)
    SS_pi = np.sum((X - p_bar[:, None] - i_bar[None, :] + grand) ** 2)
    MS_p = SS_p / (n_p - 1)
    MS_i = SS_i / (n_i - 1)
    MS_pi = SS_pi / ((n_p - 1) * (n_i - 1))
    # Random-effects variance components
    var_p = max((MS_p - MS_pi) / n_i, 0)
    var_i = max((MS_i - MS_pi) / n_p, 0)
    var_pi = MS_pi                                        # residual + interaction
    # G / Phi with n_i items
    G = var_p / (var_p + var_pi / n_i)
    Phi = var_p / (var_p + (var_i + var_pi) / n_i)
    return {"var_person": float(var_p),
            "var_item": float(var_i),
            "var_residual": float(var_pi),
            "G_coefficient": float(G),
            "Phi_coefficient": float(Phi),
            "n_persons": int(n_p), "n_items": int(n_i),
            "method": "Generalizability p x i (one-facet crossed)"}


def d_study_forecast(g_fit: dict, n_items_new: int) -> dict:
    """Forecast G and Phi for a different number of items."""
    v_p, v_i, v_pi = g_fit["var_person"], g_fit["var_item"], g_fit["var_residual"]
    G = v_p / (v_p + v_pi / n_items_new)
    Phi = v_p / (v_p + (v_i + v_pi) / n_items_new)
    return {"n_items_new": int(n_items_new),
            "G": float(G), "Phi": float(Phi)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_p, n_i = 100, 8
    person_true = rng.normal(0, 1.5, n_p)
    item_bias = rng.normal(0, 0.5, n_i)
    X = person_true[:, None] + item_bias[None, :] + rng.normal(0, 0.7, (n_p, n_i))

    r = g_theory_pi(X)
    print("=== G-theory p x i, n_p=100, n_i=8 ===")
    for k, v in r.items():
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
        else: print(f"  {k}: {v}")

    print("\n=== D-study: predicted reliability with more items ===")
    for k in (4, 8, 16, 32):
        d = d_study_forecast(r, k)
        print(f"  n_items = {k:3d}   G = {d['G']:.4f}   Phi = {d['Phi']:.4f}")

    print("\n--- library cross-check (R gtheory or gtheory package) ---")

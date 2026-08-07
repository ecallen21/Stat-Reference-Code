"""Hierarchical time-series forecasting (Reference §13.37).

Multiple time series are related by a summation hierarchy:
    Total  ->  A + B
    A      ->  A1 + A2
    B      ->  B1 + B2 + B3
Forecasts at every level must AGGREGATE COHERENTLY.  Independent forecasts
at each level rarely do; reconciliation forces consistency.

Summing matrix S (m x n_bottom) encodes the aggregation:
    Total row = 1's summed over all bottom series
    Aggregate rows = 1's summed over their descendants
    Bottom rows    = identity

Reconciled forecast:
    y_recon = S * G * y_hat_all
where G is an m -> n_bottom mapping.  Choices of G:

Bottom-up
    G = [0 | I_bottom]  -- use only bottom-level forecasts, sum up.
    + Coherent by construction.
    - Ignores information at higher levels; noisy for bottom series.

Top-down (Gross-Sohl)
    G = [p_i * e_1^T | 0]  -- disaggregate top forecast by historical
    proportion p_i.
    + Denoises higher levels.
    - Doesn't respect bottom-level dynamics.

MinT (Minimum Trace, Wickramasuriya-Athanasopoulos-Hyndman 2019)
    G = (S^T W^-1 S)^-1 S^T W^-1
    where W is the covariance of forecast errors across ALL levels.
    Minimizes trace of the reconciled-forecast error covariance under the
    coherence constraint.  Optimal under mild conditions.

The MinT diagonal-W variant (W = diag(var errors)) is the practical default.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def build_summing_matrix(hierarchy: dict) -> tuple:
    """Build S matrix from a two-level hierarchy dict {group: [items]}.

    Returns S (m x n_bottom) and list of series names (aggregate then bottom).
    """
    groups = list(hierarchy.keys())
    bottom = [item for g in groups for item in hierarchy[g]]
    n_b = len(bottom); n_g = len(groups)
    # Row 0 = total (sum of all bottom), then groups, then bottom identity
    S = np.zeros((1 + n_g + n_b, n_b))
    S[0, :] = 1.0  # total
    for i, g in enumerate(groups):
        idx = [bottom.index(item) for item in hierarchy[g]]
        S[1 + i, idx] = 1.0
    S[1 + n_g:, :] = np.eye(n_b)
    names = ["total"] + groups + bottom
    return S, names


def bottom_up(y_bottom, S) -> np.ndarray:
    """Aggregate bottom-level forecasts up through the hierarchy."""
    return S @ np.asarray(y_bottom, dtype=float)


def top_down(y_top: float, proportions, S) -> np.ndarray:
    """Disaggregate a top-level forecast by fixed proportions and sum up."""
    p = np.asarray(proportions, dtype=float); p = p / p.sum()
    y_bottom = y_top * p
    return S @ y_bottom


def mint_reconciliation(y_hat_all, S, W=None) -> dict:
    """MinT reconciliation.  W = covariance of base-forecast errors (m x m).

    If W is None, use diagonal identity (OLS reconciliation).
    """
    y_hat_all = np.asarray(y_hat_all, dtype=float)
    m, n_b = S.shape
    if W is None:
        W = np.eye(m)
    Winv = np.linalg.pinv(W)
    G = np.linalg.pinv(S.T @ Winv @ S) @ S.T @ Winv
    y_bottom_recon = G @ y_hat_all
    y_recon = S @ y_bottom_recon
    return {"reconciled": y_recon,
            "bottom_reconciled": y_bottom_recon,
            "method": "MinT reconciliation" + (" (diag-W)" if W is None or np.allclose(W, np.diag(np.diag(W))) else "")}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Hierarchy: {Region A: [A1, A2], Region B: [B1, B2, B3]}
    hierarchy = {"A": ["A1", "A2"], "B": ["B1", "B2", "B3"]}
    S, names = build_summing_matrix(hierarchy)
    print(f"=== Summing matrix S ({S.shape[0]} x {S.shape[1]}) ===")
    print(f"  series order: {names}")
    print(S)

    # Independent (incoherent) base forecasts
    y_true_bottom = np.array([100, 120, 80, 90, 110], dtype=float)
    # Base forecasts with different noise at each level
    err = rng.normal(0, 10, S.shape[0])
    y_hat_all = S @ y_true_bottom + err
    print(f"\n=== Incoherent base forecasts vs coherent truth ===")
    for i, n in enumerate(names):
        truth = (S @ y_true_bottom)[i]
        print(f"  {n:5s}: forecast = {y_hat_all[i]:7.2f}  truth = {truth:7.2f}")

    print("\n=== Bottom-up (ignores upper base forecasts) ===")
    y_bu = bottom_up(y_hat_all[-5:], S)  # last 5 are the bottom series
    for i, n in enumerate(names):
        print(f"  {n:5s}: {y_bu[i]:7.2f}")

    print("\n=== Top-down (equal proportions) ===")
    y_td = top_down(y_hat_all[0], proportions=[0.2, 0.2, 0.2, 0.2, 0.2], S=S)
    for i, n in enumerate(names):
        print(f"  {n:5s}: {y_td[i]:7.2f}")

    print("\n=== MinT (diag-W = residual variance per series) ===")
    W = np.diag(np.array([100, 50, 50, 30, 30, 30, 30, 30]))  # top-of-hierarchy series noisier
    y_mint = mint_reconciliation(y_hat_all, S, W=W)
    for i, n in enumerate(names):
        print(f"  {n:5s}: {y_mint['reconciled'][i]:7.2f}")

    print("\n=== MSE vs coherent truth ===")
    truth = S @ y_true_bottom
    print(f"  incoherent base: {np.mean((y_hat_all - truth) ** 2):.2f}")
    print(f"  bottom-up      : {np.mean((y_bu - truth) ** 2):.2f}")
    print(f"  top-down       : {np.mean((y_td - truth) ** 2):.2f}")
    print(f"  MinT diag      : {np.mean((y_mint['reconciled'] - truth) ** 2):.2f}")

"""Log-linear models for multi-way contingency tables (Reference §8.1, §8.14).

A multi-way table of counts can be modeled as a Poisson GLM with a log link
whose predictors are indicator variables for the classifying factors and their
interactions. This lets us:

  - Test *conditional independence* structures (e.g. "A and B are independent
    given C") by fitting the corresponding hierarchical model and comparing to
    the saturated model via a likelihood-ratio (deviance) test.
  - Get expected cell counts and residuals under each model.
  - For SQUARE tables of rater agreement (row = rater 1, col = rater 2, same
    categories), fit specialized models: independence, symmetry, quasi-symmetry,
    quasi-independence -- so agreement can be decomposed beyond just kappa.

Model families implemented
--------------------------
For a 3-way table (I x J x K), the hierarchical model space includes:
    [A][B][C]           mutual independence            (no interactions)
    [AB][C]             AB dependent, both indep of C
    [AB][AC]            A separates B and C (conditional indep of B, C | A)
    [AB][AC][BC]        no 3-way interaction (all 2-way present)
    [ABC]               saturated (fits exactly)

Deviance G^2 = 2 sum n_ij log(n_ij / mu_ij) tests fit vs. saturated.

For 2-way SQUARE agreement tables, we implement:
    - independence      log mu_ij = mu + a_i + b_j          (no diagonal term)
    - quasi-independence log mu_ij = mu + a_i + b_j + delta_i * I(i == j)
    - symmetry           mu_ij = mu_ji            (fit via a shared-parameter GLM)
    - quasi-symmetry     symmetry + separate main effects
Comparisons among these decompose agreement into chance, main-effect bias, and
symmetric-error components (Agresti 2013, Ch. 10).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import itertools    # stdlib: cartesian products / combinations for building factor grids
import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from typing import Sequence    # stdlib: type hint meaning 'indexable iterable' (list / tuple / array)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # distributions, hypothesis tests, PPFs (norm, t, chi2, ttest_ind, ...)


def _poisson_irls(X, y, max_iter=100, tol=1e-10):
    """IRLS for Poisson GLM with log link (see techniques/poisson-regression)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)
    for _ in range(max_iter):
        eta = X @ beta; mu = np.exp(np.clip(eta, -500, 500))
        w = np.clip(mu, 1e-12, None)
        z = eta + (y - mu) / w
        sw = np.sqrt(w); Xw = X * sw[:, None]; zw = z * sw
        beta_new, *_ = np.linalg.lstsq(Xw, zw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new; break
        beta = beta_new
    mu = np.exp(X @ beta)
    # deviance = 2 * sum (y log(y/mu) - (y - mu))    (y=0 cells contribute 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(y > 0, np.log(np.where(y > 0, y, 1) / np.clip(mu, 1e-12, None)), 0.0)
    dev_terms = y * log_ratio - (y - mu)
    dev = 2 * float(dev_terms.sum())
    return beta, mu, dev


def _dummy_matrix(levels_per_factor: Sequence[int], interactions: Sequence[Sequence[int]]):
    """Build a design matrix for hierarchical log-linear models on the FULL
    Cartesian grid of factor combinations. Uses treatment coding (drop first level).

    ``levels_per_factor`` gives the K_j for each factor.
    ``interactions`` is a list of index-tuples like [(0,), (1,), (2,), (0,1)] meaning
    include main effects for factors 0,1,2 and the 0x1 interaction.

    Returns
    -------
    X : (N, p) design matrix with the intercept column first.
    col_names : list of column names.
    grid : (N, len(levels_per_factor)) array of factor values (integer levels
        starting at 0) -- rows are aligned with X.
    """
    ranges = [list(range(L)) for L in levels_per_factor]
    grid = np.array(list(itertools.product(*ranges)))
    N = grid.shape[0]
    columns = [np.ones(N)]
    col_names = ["intercept"]
    for term in interactions:
        # For a term like (a, b), take the cross-product of level-indicators for
        # each factor in the term (dropping level 0 for each).
        factor_dummies = []
        factor_names = []
        for f in term:
            L = levels_per_factor[f]
            dums = []; names = []
            for lvl in range(1, L):
                col = (grid[:, f] == lvl).astype(float)
                dums.append(col); names.append(f"F{f}={lvl}")
            factor_dummies.append(dums); factor_names.append(names)
        # cross-product of dummies across factors in the term
        for combo in itertools.product(*[range(len(fd)) for fd in factor_dummies]):
            col = np.ones(N)
            name_parts = []
            for f_idx, choose in enumerate(combo):
                col = col * factor_dummies[f_idx][choose]
                name_parts.append(factor_names[f_idx][choose])
            columns.append(col); col_names.append(":".join(name_parts))
    X = np.column_stack(columns)
    return X, col_names, grid


def fit_loglinear(counts, levels_per_factor, interactions) -> dict:
    """Fit a hierarchical log-linear model to a multi-way count table.

    Parameters
    ----------
    counts : flattened count array aligned with the Cartesian grid produced by
        ``levels_per_factor`` (row-major, e.g. via ``table.flatten()``).
    levels_per_factor : sequence of factor level counts, one per axis.
    interactions : list of index-tuples describing the terms to include.

    Returns
    -------
    dict with fitted coefficients, expected counts, deviance G^2, df, and p.
    """
    X, names, grid = _dummy_matrix(levels_per_factor, interactions)
    beta, mu, dev = _poisson_irls(X, counts)
    N = X.shape[0]; p = X.shape[1]
    df = N - p
    p_val = float(stats.chi2.sf(dev, df)) if df > 0 else float("nan")
    # Pearson X^2 as well
    x2 = float(((counts - mu) ** 2 / np.clip(mu, 1e-12, None)).sum())
    return {"coefficients": dict(zip(names, beta.tolist())),
            "expected_counts": mu.tolist(),
            "deviance_G2": dev, "pearson_X2": x2,
            "df": df, "p_value_deviance": p_val,
            "terms": [tuple(t) for t in interactions],
            "n_cells": N, "n_params": p}


def compare_models(counts, levels_per_factor, terms_small, terms_large) -> dict:
    """LR test comparing a nested pair of log-linear models."""
    sm = fit_loglinear(counts, levels_per_factor, terms_small)
    lg = fit_loglinear(counts, levels_per_factor, terms_large)
    dG2 = sm["deviance_G2"] - lg["deviance_G2"]
    ddf = sm["df"] - lg["df"]
    return {"small_G2": sm["deviance_G2"], "large_G2": lg["deviance_G2"],
            "delta_G2": dG2, "delta_df": ddf,
            "p_value": float(stats.chi2.sf(dG2, ddf)) if ddf > 0 else float("nan"),
            "method": "LR test (nested log-linear models)"}


# --------------------------------------------------------------------------
# Rater agreement models on a K x K square table
# --------------------------------------------------------------------------

def _build_agreement_designs(K):
    """Build design matrices for independence, quasi-independence, quasi-symmetry."""
    # Row/col indices for each cell (row-major)
    rows = np.repeat(np.arange(K), K); cols = np.tile(np.arange(K), K)
    N = K * K
    ones = np.ones(N)
    # Independence: intercept + (K-1) row dummies + (K-1) col dummies
    def dums(ix):
        return np.column_stack([(ix == lv).astype(float) for lv in range(1, K)])
    Xi = np.column_stack([ones, dums(rows), dums(cols)])
    # Quasi-independence: independence + one delta per diagonal cell (K parameters)
    diag_ind = np.column_stack([(rows == d) & (cols == d) for d in range(K)]).astype(float)
    Xqi = np.column_stack([Xi, diag_ind])
    # Quasi-symmetry: independence + K(K-1)/2 symmetric-pair indicators
    # For each i<j, we add a column that is 1 for (i,j) and (j,i).
    sym_cols = []
    for i in range(K):
        for j in range(i + 1, K):
            col = ((rows == i) & (cols == j)) | ((rows == j) & (cols == i))
            sym_cols.append(col.astype(float))
    if sym_cols:
        Xqs = np.column_stack([Xi, np.column_stack(sym_cols)])
    else:
        Xqs = Xi
    return Xi, Xqi, Xqs


def fit_agreement_models(square_counts) -> dict:
    """Fit independence, quasi-independence, and quasi-symmetry models."""
    M = np.asarray(square_counts, dtype=float)
    K = M.shape[0]
    if K != M.shape[1]:
        raise ValueError("square_counts must be K x K")
    y = M.flatten()
    Xi, Xqi, Xqs = _build_agreement_designs(K)
    out = {}
    for name, X in (("independence", Xi),
                    ("quasi_independence", Xqi),
                    ("quasi_symmetry", Xqs)):
        _, mu, dev = _poisson_irls(X, y)
        df = X.shape[0] - X.shape[1]
        out[name] = {"G2": dev, "df": df,
                      "p_value": float(stats.chi2.sf(dev, df)) if df > 0 else float("nan"),
                      "n_params": X.shape[1]}
    # LR test of quasi-symmetry vs. quasi-independence (assesses agreement > chance)
    delta_G2 = out["quasi_independence"]["G2"] - out["quasi_symmetry"]["G2"]
    delta_df = out["quasi_independence"]["df"] - out["quasi_symmetry"]["df"]
    out["QS_vs_QI_lr"] = {"delta_G2": delta_G2, "delta_df": delta_df,
                          "p_value": float(stats.chi2.sf(delta_G2, delta_df))
                                     if delta_df > 0 else float("nan")}
    return out


def library_versions(counts_flat, levels, terms_small, terms_large, square_counts):
    import statsmodels.api as sm
    X_small, _, _ = _dummy_matrix(levels, terms_small)
    X_large, _, _ = _dummy_matrix(levels, terms_large)
    m_sm = sm.GLM(counts_flat, X_small, family=sm.families.Poisson()).fit()
    m_lg = sm.GLM(counts_flat, X_large, family=sm.families.Poisson()).fit()
    return {"statsmodels small deviance": float(m_sm.deviance),
            "statsmodels large deviance": float(m_lg.deviance),
            "delta_deviance": float(m_sm.deviance - m_lg.deviance)}


if __name__ == "__main__":
    # --- 3-way example: 2 x 2 x 2 table -------------------------------------
    # counts along the (A, B, C) grid, row-major over the Cartesian product
    counts = np.array([50, 60, 30, 40, 20, 45, 25, 35], dtype=float)  # 8 cells
    levels = [2, 2, 2]
    # Independence [A][B][C]
    indep = [(0,), (1,), (2,)]
    # All 2-way [AB][AC][BC]
    two_way = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]

    print("=== [A][B][C] mutual independence ===")
    r1 = fit_loglinear(counts, levels, indep)
    print(f"  G^2 = {r1['deviance_G2']:.4f}, df = {r1['df']}, p = {r1['p_value_deviance']:.4g}")

    print("\n=== [AB][AC][BC] no 3-way interaction ===")
    r2 = fit_loglinear(counts, levels, two_way)
    print(f"  G^2 = {r2['deviance_G2']:.4f}, df = {r2['df']}, p = {r2['p_value_deviance']:.4g}")

    print("\n=== LR test: indep vs. all-2-way ===")
    for k, v in compare_models(counts, levels, indep, two_way).items():
        print(f"  {k:14s}: {v}")

    print("\n--- library (statsmodels GLM Poisson) ---")
    for k, v in library_versions(counts, levels, indep, two_way, None).items():
        print(f"  {k}: {v}")

    # --- Rater agreement on a 4x4 square table ------------------------------
    square = np.array([
        [50,  8,  1,  0],
        [ 7, 30,  6,  1],
        [ 1,  5, 25,  4],
        [ 0,  1,  3, 20],
    ], dtype=float)
    print("\n=== Agreement models (4x4) ===")
    for k, v in fit_agreement_models(square).items():
        print(f"  {k:22s}: {v}")

"""Variable selection in regression (Reference §5.8, §5.18, §5.19, §5.36).

CLASSICAL METHODS (with warnings)
---------------------------------
- Forward stepwise   : start empty; add the predictor that improves the
                       criterion the most; stop when no addition improves it.
- Backward stepwise  : start full; drop the predictor whose removal improves
                       the criterion the most; stop when no removal improves.
- Bidirectional      : at each step, consider both adds and drops.
- Best subsets       : evaluate ALL 2^p models; report the best by criterion.

Selection criteria
  - AIC  = n * log(RSS / n) + 2 p          (preferred for prediction)
  - BIC  = n * log(RSS / n) + log(n) p     (heavier penalty; preferred for "true model")
  - adjusted R^2

A CRITICAL CAVEAT (§5.36): stepwise methods are PROBLEMATIC.
  - Inflated R^2, biased coefficient estimates, p-values that aren't valid.
  - "Conditional on the selected model" inference is no longer the inference
    you printed.
  - Modern preference: regularization (techniques/regularization) -- lasso /
    elastic net handle variable selection AND give better predictive
    performance, AND they can be CV'd.

We implement these for completeness / understanding, with the caveat front and
center in the README.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def _fit_rss(X, y):
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    return float(e @ e)


def criterion(X, y, kind: str = "AIC") -> float:
    """AIC, BIC, or adjusted R^2 (returned NEGATED so smaller is always better)."""
    n, p = np.shape(X)
    rss = _fit_rss(X, y)
    if kind == "AIC":
        return n * math.log(rss / n) + 2 * p
    if kind == "BIC":
        return n * math.log(rss / n) + math.log(n) * p
    if kind == "adj_R2":
        tss = float(((np.asarray(y) - np.mean(y)) ** 2).sum())
        r2 = 1 - rss / tss
        return -(1 - (1 - r2) * (n - 1) / (n - p))   # negate so smaller-is-better
    raise ValueError("kind must be 'AIC', 'BIC', or 'adj_R2'")


def _design(X_full, intercept: bool, idx):
    cols = []
    if intercept:
        cols.append(np.ones(X_full.shape[0]))
    cols.extend(X_full[:, j] for j in idx)
    return np.column_stack(cols) if cols else np.zeros((X_full.shape[0], 0))


def forward_stepwise(X, y, kind: str = "AIC", verbose: bool = False) -> dict:
    """Forward selection by ``kind`` (AIC / BIC / adj_R2)."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    selected = []
    remaining = list(range(p))
    best_crit = criterion(_design(X, True, []), y, kind)
    history = [{"step": 0, "selected": [], "criterion": best_crit}]
    while remaining:
        scores = []
        for j in remaining:
            cand = selected + [j]
            scores.append((criterion(_design(X, True, cand), y, kind), j))
        c_new, j_new = min(scores)
        if c_new < best_crit:
            best_crit = c_new
            selected.append(j_new); remaining.remove(j_new)
            history.append({"step": len(selected), "selected": selected.copy(),
                            "criterion": c_new, "added": j_new})
            if verbose: print(f"  + j={j_new}  -> {kind} = {c_new:.4f}")
        else:
            break
    return {"selected": selected, "best_criterion": best_crit,
            "history": history, "criterion_kind": kind}


def backward_stepwise(X, y, kind: str = "AIC", verbose: bool = False) -> dict:
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    n, p = X.shape
    selected = list(range(p))
    best_crit = criterion(_design(X, True, selected), y, kind)
    history = [{"step": 0, "selected": selected.copy(), "criterion": best_crit}]
    while selected:
        scores = []
        for j in selected:
            cand = [k for k in selected if k != j]
            scores.append((criterion(_design(X, True, cand), y, kind), j))
        c_new, j_drop = min(scores)
        if c_new < best_crit:
            best_crit = c_new
            selected.remove(j_drop)
            history.append({"step": len(history), "selected": selected.copy(),
                            "criterion": c_new, "dropped": j_drop})
            if verbose: print(f"  - j={j_drop}  -> {kind} = {c_new:.4f}")
        else:
            break
    return {"selected": selected, "best_criterion": best_crit,
            "history": history, "criterion_kind": kind}


def best_subsets(X, y, kind: str = "AIC", max_size: int | None = None) -> dict:
    """Exhaustive search over all subsets (up to ``max_size``). Feasible for p <= ~20."""
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    p = X.shape[1]
    max_size = p if max_size is None else min(max_size, p)
    best_per_size = {}
    overall_best = (math.inf, None)
    for k in range(0, max_size + 1):
        best_here = (math.inf, None)
        for subset in combinations(range(p), k):
            c = criterion(_design(X, True, subset), y, kind)
            if c < best_here[0]:
                best_here = (c, list(subset))
        best_per_size[k] = best_here
        if best_here[0] < overall_best[0]:
            overall_best = best_here
    return {"best_per_size": best_per_size,
            "overall_best_criterion": overall_best[0],
            "overall_best_subset": overall_best[1],
            "criterion_kind": kind}


def library_versions(X, y):
    # statsmodels has no built-in stepwise; mlxtend / sklearn-genetic offer some.
    # For "stepwise via AIC" the standard reference is R's stats::step.
    return {"note": "Python has no canonical stepwise; see R's stats::step or use lasso (techniques/regularization)"}


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n, p = 100, 8
    X = rng.normal(0, 1, (n, p))
    true_beta = np.array([2.0, -1.5, 0.0, 0.0, 1.0, 0.0, 0.0, -0.3])
    y = X @ true_beta + rng.normal(0, 1, n)

    print("=== Forward stepwise (AIC) ===")
    res_fwd = forward_stepwise(X, y, kind="AIC", verbose=True)
    print(f"  selected indices: {res_fwd['selected']}")

    print("\n=== Backward stepwise (AIC) ===")
    res_bwd = backward_stepwise(X, y, kind="AIC", verbose=True)
    print(f"  selected indices: {res_bwd['selected']}")

    print("\n=== Best subsets (BIC, max 5 predictors) ===")
    res_bs = best_subsets(X, y, kind="BIC", max_size=5)
    print(f"  overall best subset (BIC): {res_bs['overall_best_subset']}  "
          f"BIC = {res_bs['overall_best_criterion']:.3f}")
    for size, (c, subset) in res_bs["best_per_size"].items():
        print(f"  size {size}: subset {subset}  BIC = {c:.3f}")

    print(f"\n  true non-zero indices: {[i for i, b in enumerate(true_beta) if b != 0]}")
    print("\n  See techniques/regularization for the modern preferred alternative (lasso).")

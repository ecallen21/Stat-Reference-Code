"""Coarsened Exact Matching (Reference Sec 15.10).

Iacus-King-Porro 2012.  Discretise ("coarsen") continuous covariates
into bins, then EXACT MATCH treated to control units within each
coarsened cell.  Compute weights so within-cell control units sum
to treated count.

Advantages:
  * Guarantees perfect covariate balance on the coarsened scale.
  * Non-parametric (no PS model to misspecify).
  * Straightforward diagnostic: L1 imbalance measure.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def coarsen(X, breaks_list):
    """Bin each column of X using its provided breaks."""
    idx = np.zeros_like(X, dtype=int)
    for j in range(X.shape[1]):
        idx[:, j] = np.digitize(X[:, j], breaks_list[j])
    return idx


def cem_weights(X, T, breaks_list):
    """Compute CEM weights: 1 for treated within matched cells;
    (n_t_cell / n_c_cell) * (n_t_total_matched / n_c_total_matched) for controls;
    0 for unmatched units.
    """
    strata = coarsen(X, breaks_list)
    id_tuples = [tuple(row.tolist()) for row in strata]
    unique_ids = set(id_tuples)
    w = np.zeros(len(T), dtype=float)
    id_arr = np.array(id_tuples, dtype=object)
    for s in unique_ids:
        mask = np.array([t == s for t in id_tuples])
        n_t = ((mask) & (T == 1)).sum()
        n_c = ((mask) & (T == 0)).sum()
        if n_t == 0 or n_c == 0:
            continue                     # unmatched cell
        w[(mask) & (T == 1)] = 1.0
        w[(mask) & (T == 0)] = n_t / n_c
    # Rescale controls so overall weighted count of controls matches treated
    w_t_total = w[T == 1].sum(); w_c_total = w[T == 0].sum()
    if w_c_total > 0:
        w[T == 0] *= w_t_total / w_c_total
    return w


def att_from_weights(y, T, w):
    return float(((w * T * y).sum() / (w * T).sum()
                  - (w * (1 - T) * y).sum() / (w * (1 - T)).sum()))


if __name__ == "__main__":
    print("=== Coarsened Exact Matching ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(0, 1, (n, 2))
    logit_T = 0.6 * X[:, 0] + 0.4 * X[:, 1]
    T = (rng.random(n) < 1 / (1 + np.exp(-logit_T))).astype(int)
    # True ATT = 0.5
    y = 1.0 + 0.5 * T + 0.6 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(0, 1, n)

    breaks = [np.array([-2, -1, 0, 1, 2]) for _ in range(X.shape[1])]
    w = cem_weights(X, T, breaks)

    print(f"  Matched treated units: {int((w[T == 1] > 0).sum())} / {int((T == 1).sum())}")
    print(f"  Matched control units: {int((w[T == 0] > 0).sum())} / {int((T == 0).sum())}")

    naive = float(y[T == 1].mean() - y[T == 0].mean())
    cem_att = att_from_weights(y, T, w)
    print(f"\n  Naive difference = {naive:+.3f}   (biased)")
    print(f"  CEM ATT estimate = {cem_att:+.3f}   (true = 0.5)\n")

    print("--- library cross-check (R MatchIt::matchit(method='cem'), cem; Python custom) ---")

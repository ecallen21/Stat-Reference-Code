"""Latin square design + ANOVA (Reference §16.6).

Experimental design that blocks two nuisance factors simultaneously with
FEWER runs than a full factorial:

    row block:    i = 1..k     (e.g. day of experiment)
    col block:    j = 1..k     (e.g. lab technician)
    treatment:    l(i, j)      arranged so each treatment appears
                                exactly once in every row and column.

Example k = 3 Latin square:
    A B C
    B C A
    C A B

Only k^2 runs (vs k^3 for a full three-factor factorial), at the cost of
assuming NO INTERACTIONS between the three factors.

ANOVA model:
    y_ijl = mu + rho_i + gamma_j + tau_l + eps_ijl
        SS_total = SS_row + SS_col + SS_treatment + SS_error
        df_total = k^2 - 1
        df_row = df_col = df_treatment = k - 1
        df_error = (k - 1)(k - 2)

Extensions: Graeco-Latin (superimpose two orthogonal Latin squares to
block a fourth factor), replicated Latin square.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)
from scipy import stats    # SciPy statistical distributions (norm, t, chi2, f) and tests


def cyclic_latin_square(k: int) -> np.ndarray:
    """Standard cyclic Latin square of order k."""
    return np.array([[(i + j) % k for j in range(k)] for i in range(k)])


def randomized_latin_square(k: int, seed: int = 0) -> np.ndarray:
    """Latin square with rows / columns / treatments randomized."""
    rng = np.random.default_rng(seed)
    ls = cyclic_latin_square(k)
    row_perm = rng.permutation(k); col_perm = rng.permutation(k); trt_perm = rng.permutation(k)
    ls = ls[row_perm][:, col_perm]
    ls = np.vectorize(lambda t: trt_perm[t])(ls)
    return ls


def latin_square_anova(y, row_idx, col_idx, trt_idx) -> dict:
    """Fit Latin-square ANOVA and return the sums-of-squares table."""
    y = np.asarray(y, dtype=float)
    row_idx = np.asarray(row_idx); col_idx = np.asarray(col_idx); trt_idx = np.asarray(trt_idx)
    k = len(np.unique(row_idx)); n = len(y)
    y_bar = y.mean()
    def ss_from(idx):
        levels = np.unique(idx); ss = 0.0
        for lvl in levels:
            sel = idx == lvl
            ss += sel.sum() * (y[sel].mean() - y_bar) ** 2
        return ss
    SS_row = ss_from(row_idx)
    SS_col = ss_from(col_idx)
    SS_trt = ss_from(trt_idx)
    SS_total = float(np.sum((y - y_bar) ** 2))
    SS_error = SS_total - SS_row - SS_col - SS_trt
    df_row = df_col = df_trt = k - 1
    df_error = (k - 1) * (k - 2)
    if df_error <= 0:
        return {"error": "need k >= 3 for non-zero error df"}
    MS_trt = SS_trt / df_trt; MS_error = SS_error / df_error
    F_trt = MS_trt / MS_error
    p_trt = float(stats.f.sf(F_trt, df_trt, df_error))
    return {"SS_row": SS_row, "SS_col": SS_col, "SS_treatment": SS_trt,
            "SS_error": SS_error, "SS_total": SS_total,
            "MS_treatment": MS_trt, "MS_error": MS_error,
            "F_treatment": F_trt, "df_treatment": df_trt, "df_error": df_error,
            "p_value_treatment": p_trt,
            "k": int(k),
            "method": "Latin-square ANOVA"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    k = 5
    ls = randomized_latin_square(k, seed=0)
    print(f"=== Randomized Latin square, k = {k} ===")
    print(ls)

    # Simulate response: row_effect + col_effect + trt_effect (treatment matters)
    row_eff = np.array([0, 0.5, -0.2, 0.3, -0.1])
    col_eff = np.array([-0.1, 0.2, 0.4, -0.2, 0.1])
    trt_eff = np.array([0, 0.5, 1.0, 1.5, 2.0])
    y = np.zeros(k * k)
    row_idx = np.zeros(k * k, dtype=int); col_idx = np.zeros(k * k, dtype=int)
    trt_idx = np.zeros(k * k, dtype=int)
    idx = 0
    for i in range(k):
        for j in range(k):
            t = ls[i, j]
            y[idx] = 10 + row_eff[i] + col_eff[j] + trt_eff[t] + rng.normal(0, 0.3)
            row_idx[idx] = i; col_idx[idx] = j; trt_idx[idx] = t
            idx += 1

    r = latin_square_anova(y, row_idx, col_idx, trt_idx)
    print(f"\n=== Latin-square ANOVA table ===")
    print(f"  SS_row       = {r['SS_row']:.3f}  df = {r['k'] - 1}")
    print(f"  SS_col       = {r['SS_col']:.3f}  df = {r['k'] - 1}")
    print(f"  SS_treatment = {r['SS_treatment']:.3f}  df = {r['df_treatment']}")
    print(f"  SS_error     = {r['SS_error']:.3f}  df = {r['df_error']}")
    print(f"  F treatment  = {r['F_treatment']:.3f}   p = {r['p_value_treatment']:.4f}")

"""Split-plot design ANOVA (Reference §18.x extra).

Design:
  * A: whole-plot factor (harder to change; e.g. irrigation type)
  * B: subplot factor (easier to change; e.g. fertiliser)
  * Whole-plots (blocks / replicates) contain subplots that share the same A level.

Two error strata:
  * whole-plot error MSE_WP  = SS(WP within A) / df_WP
    (tests A: F_A = MS_A / MSE_WP)
  * subplot error MSE_SP     = SS(residual) / df_SP
    (tests B and A x B: F_B = MS_B / MSE_SP, F_AB = MS_AB / MSE_SP)

Naive one-error ANOVA (treating all subplots as independent) inflates the
test for A because the whole-plot variance is ignored.

Fit here by direct decomposition of sums of squares.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (log, sqrt)

import numpy as np    # numerical arrays + linear algebra
from scipy.stats import f as f_dist    # F distribution for p-values


def split_plot_anova(y, A, WP, B) -> dict:
    """A = whole-plot factor level per obs; WP = whole-plot ID; B = subplot factor."""
    y = np.asarray(y, dtype=float)
    A = np.asarray(A); WP = np.asarray(WP); B = np.asarray(B)
    N = len(y); grand = y.mean()

    def _lvls(x):
        return sorted(np.unique(x).tolist())
    lA, lB, lWP = _lvls(A), _lvls(B), _lvls(WP)
    a = len(lA); b = len(lB); wp = len(lWP)
    # replicates per A-level: each WP is nested in one A
    # For a balanced split-plot: r = wp / a
    r = wp // a

    # sums of squares
    SS_A = sum(r * b * (y[A == aval].mean() - grand) ** 2 for aval in lA)
    SS_B = sum(a * r * (y[B == bval].mean() - grand) ** 2 for bval in lB)
    SS_AB = 0.0
    for aval in lA:
        for bval in lB:
            m_ab = y[(A == aval) & (B == bval)].mean()
            m_a = y[A == aval].mean()
            m_b = y[B == bval].mean()
            SS_AB += r * (m_ab - m_a - m_b + grand) ** 2
    # whole-plot error: SS(WP within A) - SS(A)
    SS_WP = sum(b * (y[WP == w].mean() - grand) ** 2 for w in lWP)
    SS_WPE = SS_WP - SS_A
    # subplot residual = total - all above
    SS_T = ((y - grand) ** 2).sum()
    SS_SP = SS_T - SS_A - SS_WPE - SS_B - SS_AB

    df_A = a - 1
    df_WPE = a * (r - 1)
    df_B = b - 1
    df_AB = (a - 1) * (b - 1)
    df_SP = a * (r - 1) * (b - 1)

    MS_A = SS_A / df_A
    MS_WPE = SS_WPE / df_WPE
    MS_B = SS_B / df_B
    MS_AB = SS_AB / df_AB
    MS_SP = SS_SP / df_SP

    F_A = MS_A / MS_WPE
    F_B = MS_B / MS_SP
    F_AB = MS_AB / MS_SP

    return {"table": [
        {"source": "A (whole-plot)",  "df": df_A,   "SS": SS_A,   "MS": MS_A,   "F": F_A,
         "p": float(1 - f_dist.cdf(F_A, df_A, df_WPE)), "error": "MSE_WP"},
        {"source": "WP within A (error a)", "df": df_WPE, "SS": SS_WPE, "MS": MS_WPE},
        {"source": "B (subplot)",     "df": df_B,   "SS": SS_B,   "MS": MS_B,   "F": F_B,
         "p": float(1 - f_dist.cdf(F_B, df_B, df_SP)),  "error": "MSE_SP"},
        {"source": "A x B",           "df": df_AB,  "SS": SS_AB,  "MS": MS_AB,  "F": F_AB,
         "p": float(1 - f_dist.cdf(F_AB, df_AB, df_SP)), "error": "MSE_SP"},
        {"source": "Residual (error b)", "df": df_SP, "SS": SS_SP, "MS": MS_SP},
    ],
    "method": "Balanced split-plot ANOVA"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = 3; r = 4; b = 4
    A = np.repeat(np.arange(a), r * b)
    WP = np.repeat(np.arange(a * r), b)
    B = np.tile(np.arange(b), a * r)
    N = a * r * b

    # true effects
    A_eff = np.array([-1.0, 0.0, 1.5])
    B_eff = np.array([0.0, 0.5, 1.0, 1.5])
    AB_eff = np.zeros((a, b))                             # no interaction
    # whole-plot random effect (shared within WP)
    wp_re = rng.normal(scale=1.2, size=a * r)
    y = np.zeros(N)
    for i in range(N):
        y[i] = 10 + A_eff[A[i]] + B_eff[B[i]] + AB_eff[A[i], B[i]] + wp_re[WP[i]] \
                + rng.normal(scale=0.5)

    res = split_plot_anova(y, A, WP, B)
    print("=== Split-plot ANOVA (a=3, r=4 whole-plot reps, b=4 subplots each) ===")
    print(f"  {'source':>25}  {'df':>3}  {'SS':>10}  {'MS':>8}  {'F':>6}  {'p':>7}")
    for row in res["table"]:
        F = row.get("F", "")
        p = row.get("p", "")
        Fs = f"{F:.2f}" if F != "" else "  "
        ps = f"{p:.3f}" if p != "" else "  "
        print(f"  {row['source']:>25}  {row['df']:>3}  {row['SS']:>10.3f}  "
              f"{row['MS']:>8.3f}  {Fs:>6}  {ps:>7}")

    # contrast: naive one-error ANOVA (treats all N obs as independent)
    from scipy.stats import f as _f
    # SS_error_naive = SS_T - SS_A - SS_B - SS_AB
    SS_T = ((y - y.mean()) ** 2).sum()
    SS_naive = SS_T - res["table"][0]["SS"] - res["table"][2]["SS"] - res["table"][3]["SS"]
    df_naive = N - a - b + 1 - (a - 1) * (b - 1)
    MS_naive = SS_naive / df_naive
    F_A_naive = res["table"][0]["MS"] / MS_naive
    p_A_naive = 1 - _f.cdf(F_A_naive, res["table"][0]["df"], df_naive)
    print(f"\n  naive one-error F for A = {F_A_naive:.2f}, p = {p_A_naive:.4f}   "
          f"(compare split-plot p = {res['table'][0]['p']:.4f})")

    print("\n--- library cross-check (R lme4::lmer / lmerTest / afex::aov_car) ---")

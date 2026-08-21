"""2x2 crossover design analysis (Reference §18.x extra).

Each subject receives BOTH treatments (A and B) in one of two sequences:
  * Sequence 1: A in Period 1, B in Period 2
  * Sequence 2: B in Period 1, A in Period 2

Grizzle (1965) two-sample t-test procedure:

  Carryover test:  compare within-subject SUMS  (Y_1 + Y_2) across sequences.
                    If sequences differ in sum, carryover from period 1 to 2.
  Treatment test:  compare within-subject DIFFERENCES (Y_1 - Y_2) across sequences.
                    Under no carryover this tests the treatment effect.
  Period test:     compare period means within-subject to check period effect.

Under no carryover, the treatment test uses within-subject differences and
therefore removes between-subject variability, giving high power.  Carryover
is untestable without a washout or a parallel-arm supplement.
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation

import math    # stdlib: scalar math (sqrt)

import numpy as np    # numerical arrays + linear algebra
from scipy.stats import t as t_dist    # Student t distribution


def _welch_t(x1, x2):
    m1, m2 = x1.mean(), x2.mean()
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    n1, n2 = len(x1), len(x2)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 0.0, float("inf"), 1.0
    tstat = (m1 - m2) / se
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1) + 1e-12)
    p = 2 * (1 - t_dist.cdf(abs(tstat), df))
    return tstat, df, p


def crossover_2x2(y1_seq1, y2_seq1, y1_seq2, y2_seq2) -> dict:
    """y1, y2 = period-1 and period-2 outcomes; seq1 = AB, seq2 = BA."""
    y1_seq1 = np.asarray(y1_seq1, dtype=float); y2_seq1 = np.asarray(y2_seq1, dtype=float)
    y1_seq2 = np.asarray(y1_seq2, dtype=float); y2_seq2 = np.asarray(y2_seq2, dtype=float)

    # within-subject sums and differences per sequence
    sum_seq1 = y1_seq1 + y2_seq1
    sum_seq2 = y1_seq2 + y2_seq2
    diff_seq1 = y1_seq1 - y2_seq1                          # (A - B) for seq1
    diff_seq2 = y1_seq2 - y2_seq2                          # (B - A) for seq2
    # treatment effect (A - B) estimate: mean of [diff_seq1 - diff_seq2] / 2
    trt_effect = 0.5 * (diff_seq1.mean() - diff_seq2.mean())

    # carryover test on sums
    t_c, df_c, p_c = _welch_t(sum_seq1, sum_seq2)
    # treatment test on differences (Grizzle two-sample t)
    t_t, df_t, p_t = _welch_t(diff_seq1, diff_seq2)
    # period test: (mean(diff_seq1) + mean(diff_seq2)) / 2 estimates P1 - P2 with treatment cancelling
    n1, n2 = len(diff_seq1), len(diff_seq2)
    v1, v2 = diff_seq1.var(ddof=1), diff_seq2.var(ddof=1)
    period_effect = 0.5 * (diff_seq1.mean() + diff_seq2.mean())
    se_p = 0.5 * math.sqrt(v1 / n1 + v2 / n2)
    t_p = period_effect / se_p if se_p > 0 else 0.0
    df_p = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1) + 1e-12)
    p_p = 2 * (1 - t_dist.cdf(abs(t_p), df_p))
    return {"treatment_effect_A_minus_B": float(trt_effect),
            "treatment_test": {"t": float(t_t), "df": float(df_t), "p": float(p_t)},
            "carryover_test": {"t": float(t_c), "df": float(df_c), "p": float(p_c)},
            "period_test":    {"t": float(t_p), "df": float(df_p),
                                "p": float(p_p),
                                "estimate_period1_minus_period2": float(period_effect)},
            "method": "Grizzle 2x2 crossover t-tests"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 30                                                # subjects per sequence
    A_effect = 3.0                                        # true A - B
    period_effect = 0.5                                    # true P1 - P2
    subject_re_sd = 4.0
    resid_sd = 1.5

    # Seq 1: A then B
    subj1 = rng.normal(scale=subject_re_sd, size=n)
    y1_seq1 = 10 + subj1 + A_effect + period_effect / 2 + rng.normal(scale=resid_sd, size=n)
    y2_seq1 = 10 + subj1              - period_effect / 2 + rng.normal(scale=resid_sd, size=n)

    # Seq 2: B then A
    subj2 = rng.normal(scale=subject_re_sd, size=n)
    y1_seq2 = 10 + subj2              + period_effect / 2 + rng.normal(scale=resid_sd, size=n)
    y2_seq2 = 10 + subj2 + A_effect - period_effect / 2 + rng.normal(scale=resid_sd, size=n)

    res = crossover_2x2(y1_seq1, y2_seq1, y1_seq2, y2_seq2)
    print(f"=== 2x2 crossover analysis (n={n}/sequence; true A-B = {A_effect}, "
          f"period = {period_effect}) ===")
    print(f"  treatment effect (A - B) = {res['treatment_effect_A_minus_B']:+.3f}   "
          f"true = {A_effect}")
    for name, block in [("treatment", "treatment_test"),
                        ("carryover", "carryover_test"),
                        ("period",    "period_test")]:
        b = res[block]
        print(f"  {name:>10}: t = {b['t']:+.3f}  df = {b['df']:.1f}  p = {b['p']:.4f}")

    # contrast: naive between-subject t comparing periods
    from scipy.stats import ttest_ind
    A_pooled = np.concatenate([y1_seq1, y2_seq2])
    B_pooled = np.concatenate([y2_seq1, y1_seq2])
    t_naive, p_naive = ttest_ind(A_pooled, B_pooled, equal_var=False)
    print(f"\n  naive between-subject A-B t = {t_naive:+.2f}, p = {p_naive:.4f}   "
          f"(ignores within-subject correlation)")

    print("\n--- library cross-check (R Crossover / geepack::geeglm) ---")

"""Effect size measures (Reference §1.6 and §1.25).

Effect sizes quantify the *magnitude* of a difference or association -- always
report them (with CIs) alongside p-values.

Standardized mean differences (two groups)
  - Cohen's d  : (mean1 - mean2) / s_pooled         (pooled SD assumes equal variances)
  - Hedges' g  : Cohen's d * J(df)                   (small-sample bias correction)
  - Glass's d  : (mean1 - mean2) / sd_control        (use when variances differ a lot)

Nonparametric (ordinal / rank based)
  - Cliff's delta       : P(X1 > X2) - P(X1 < X2)     in [-1, 1]
  - rank-biserial r      : 1 - 2U / (n1 n2)            (from Mann-Whitney U; = Cliff's delta)

ANOVA / variance-explained (k groups)
  - eta-squared eta^2    : SS_between / SS_total
  - omega-squared        : (SS_b - (k-1) MS_within) / (SS_total + MS_within)   (less biased)
  - Cohen's f            : sqrt(eta^2 / (1 - eta^2))

Conventional benchmarks (Cohen 1988; context always trumps these):
  d:  0.2 / 0.5 / 0.8   |  r: 0.1 / 0.3 / 0.5  |  eta^2: 0.01 / 0.06 / 0.14  |  f: 0.1 / 0.25 / 0.4
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _mean(x):
    return sum(x) / len(x)


def _var(x, ddof=1):
    m = _mean(x)
    return sum((v - m) ** 2 for v in x) / (len(x) - ddof)


def cohens_d(x1: Sequence[float], x2: Sequence[float]) -> float:
    n1, n2 = len(x1), len(x2)
    sp2 = ((n1 - 1) * _var(x1) + (n2 - 1) * _var(x2)) / (n1 + n2 - 2)
    return (_mean(x1) - _mean(x2)) / math.sqrt(sp2)


def hedges_g(x1: Sequence[float], x2: Sequence[float]) -> float:
    n1, n2 = len(x1), len(x2)
    df = n1 + n2 - 2
    j = 1.0 - 3.0 / (4.0 * df - 1.0)  # bias-correction factor J(df)
    return cohens_d(x1, x2) * j


def glass_delta(treatment: Sequence[float], control: Sequence[float]) -> float:
    return (_mean(treatment) - _mean(control)) / math.sqrt(_var(control))


def cliffs_delta(x1: Sequence[float], x2: Sequence[float]) -> float:
    gt = lt = 0
    for a in x1:
        for b in x2:
            if a > b:
                gt += 1
            elif a < b:
                lt += 1
    return (gt - lt) / (len(x1) * len(x2))


def rank_biserial_from_u(u1: float, n1: int, n2: int) -> float:
    """rank-biserial correlation from the Mann-Whitney U1 of group 1 (equals Cliff's delta).

    U1 here is the statistic scipy returns as ``stats.mannwhitneyu(x1, x2).statistic``:
    the count of (x1 > x2) pairs (ties count 0.5). Then r_rb = 2*U1/(n1 n2) - 1.
    """
    return 2.0 * u1 / (n1 * n2) - 1.0


def eta_squared_oneway(groups: Sequence[Sequence[float]]):
    """One-way ANOVA effect sizes. Returns dict with eta^2, omega^2, Cohen's f."""
    allvals = [v for g in groups for v in g]
    grand = _mean(allvals)
    N = len(allvals)
    k = len(groups)
    ss_between = sum(len(g) * (_mean(g) - grand) ** 2 for g in groups)
    ss_total = sum((v - grand) ** 2 for v in allvals)
    ss_within = ss_total - ss_between
    ms_within = ss_within / (N - k)
    eta2 = ss_between / ss_total
    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    f = math.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else float("inf")
    return {"eta_squared": eta2, "omega_squared": omega2, "cohens_f": f}


def interpret(value: float, kind: str) -> str:
    """Rough Cohen-style verbal label. kind in {'d', 'r', 'eta2', 'f'}."""
    cuts = {
        "d": [(0.2, "negligible"), (0.5, "small"), (0.8, "medium"), (math.inf, "large")],
        "r": [(0.1, "negligible"), (0.3, "small"), (0.5, "medium"), (math.inf, "large")],
        "eta2": [(0.01, "negligible"), (0.06, "small"), (0.14, "medium"), (math.inf, "large")],
        "f": [(0.1, "negligible"), (0.25, "small"), (0.4, "medium"), (math.inf, "large")],
    }[kind]
    v = abs(value)
    for thresh, label in cuts:
        if v < thresh:
            return label
    return "large"


def library_versions(x1, x2):
    out = {}
    try:
        import pingouin as pg

        out["Cohen's d (pingouin)"] = float(pg.compute_effsize(x1, x2, eftype="cohen"))
        out["Hedges' g (pingouin)"] = float(pg.compute_effsize(x1, x2, eftype="hedges"))
        out["Cliff's delta (pingouin)"] = float(pg.compute_effsize(x1, x2, eftype="CLES")) * 2 - 1
    except ImportError:
        out["note"] = "install 'pingouin' for pg.compute_effsize(); scipy used below"
    from scipy import stats

    u, _ = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    out["rank-biserial from scipy U"] = rank_biserial_from_u(u, len(x1), len(x2))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    a = np.round(rng.normal(105, 15, 30), 1).tolist()
    b = np.round(rng.normal(100, 15, 28), 1).tolist()
    print(f"group A: n={len(a)} mean={np.mean(a):.2f}   group B: n={len(b)} mean={np.mean(b):.2f}\n")
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    cd = cliffs_delta(a, b)
    print("--- from scratch (two groups) ---")
    print(f"Cohen's d      : {d:.4f}  ({interpret(d, 'd')})")
    print(f"Hedges' g      : {g:.4f}  ({interpret(g, 'd')})")
    print(f"Glass's delta  : {glass_delta(a, b):.4f}")
    print(f"Cliff's delta  : {cd:.4f}  ({interpret(cd, 'r')})")
    print("\n--- from scratch (one-way ANOVA, 3 groups) ---")
    c = np.round(rng.normal(110, 15, 25), 1).tolist()
    es = eta_squared_oneway([a, b, c])
    print(f"eta^2          : {es['eta_squared']:.4f}  ({interpret(es['eta_squared'], 'eta2')})")
    print(f"omega^2        : {es['omega_squared']:.4f}")
    print(f"Cohen's f      : {es['cohens_f']:.4f}  ({interpret(es['cohens_f'], 'f')})")
    print("\n--- library ---")
    for k, v in library_versions(a, b).items():
        print(f"{k:30s}: {v}")

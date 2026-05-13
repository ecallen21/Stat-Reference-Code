# Goodman–Kruskal γ, λ, τ, Somers' D, Kendall's τ-b on Contingency Tables (Reference §4.11, §4.12)

Two families here:

1. **Ordinal-pair-based** (γ, Somers' D, Kendall's τ-b) — sensitive to category order.
2. **Nominal PRE-based** (Goodman–Kruskal λ and τ) — order-free; measure how much knowing X reduces the *error* / *variance* in predicting Y.

## Ordinal pair-based measures

Built from concordant / discordant pair counts:

- `C` = # concordant pairs (both variables agree on which observation is higher)
- `D` = # discordant pairs (variables disagree)
- `T_x` = # pairs tied on X (rows) only
- `T_y` = # pairs tied on Y (columns) only

| Measure | Formula | Treats ties as | Symmetric? | Range |
|---------|---------|-----------------|------------|-------|
| **Goodman–Kruskal γ** | `(C − D) / (C + D)` | invisible | yes | `[−1, 1]` |
| **Somers' D(y \| x)** | `(C − D) / (C + D + T_y)` | penalizes only ties on Y | **no** | `[−1, 1]` |
| **Somers' D(x \| y)** | `(C − D) / (C + D + T_x)` | penalizes only ties on X | no | `[−1, 1]` |
| **Kendall's τ-b** | `(C − D) / √((C + D + T_x)(C + D + T_y))` | penalizes both | yes | `[−1, 1]` |

γ is the most "optimistic," τ-b the most "pessimistic"; Somers' D is the **asymmetric** version — appropriate when one variable is conceptually a predictor and the other an outcome.

## When to use which

- **γ** — quick descriptive number; less defensible when there are many ties.
- **Somers' D(y | x)** — the standard *predictive* association measure for ordinal logistic regression; e.g. SAS `PROC LOGISTIC` reports it as a model-discrimination diagnostic (equivalent to `2·AUC − 1` for binary outcomes).
- **τ-b** — the conventional ordinal-table equivalent of Kendall's τ (`techniques/kendalls-tau`); use when you'd otherwise reach for τ.

All three give the same answer when there are **no ties** on the variable each penalizes.

## Nominal PRE measures

**Goodman–Kruskal λ** (lambda) — "mode rule" proportional reduction in error:
- `P_e   = 1 − max_y(n_{·y}/N)`  → error rate of predicting Y without knowing X (always guess the modal category).
- `P_{e|X} = (N − Σ_x max_y n_{xy}) / N` → error given X (mode rule per row).
- `λ(y | x) = (P_e − P_{e|X}) / P_e` ∈ `[0, 1]`.

**Goodman–Kruskal τ** (tau) — variance-based proportional reduction:
- `V_y      = 1 − Σ_y (n_{·y}/N)²`        (Gini variance of Y)
- `V_{y|X}  = 1 − Σ_x (n_{x·}/N)·Σ_y (n_{xy}/n_{x·})²`
- `τ(y | x) = (V_y − V_{y|X}) / V_y` ∈ `[0, 1]`.

Both are **asymmetric** — direction matters. The naming overlap with Kendall's τ is unfortunate: **Goodman–Kruskal τ ≠ Kendall's τ**, despite sometimes being printed under the same heading in software.

When to pick which:
- **λ** is intuitive ("how much does knowing X reduce my mode-rule mistakes?") but can be zero even with real association if the modal Y is the same in every X stratum.
- **τ** uses the full distribution, not just the mode — more sensitive, recommended over λ in most cases.

## Counting pairs from a table

Closed form (used by both implementations):
- `C = Σ n_{ij} · Σ_{r > i, s > j} n_{rs}` — strictly upper-right neighbors.
- `D = Σ n_{ij} · Σ_{r > i, s < j} n_{rs}` — strictly lower-right neighbors.
- `T_x = Σ_{i, j < j'} n_{ij} · n_{ij'}` — same row, different columns.
- `T_y = Σ_{j, i < i'} n_{ij} · n_{i'j}` — same column, different rows.

This is `O((rc)²)` for a small table — fine.

## Files
- `python/goodman_kruskal_somers.py` — from-scratch `C / D / T_x / T_y` counts and all four measures; the demo on a 3×3 table shows γ > Somers' D > τ-b as expected.
- `r/goodman_kruskal_somers.R` — from-scratch + `DescTools::GoodmanKruskalGamma`, `DescTools::SomersDelta` (with `direction = "column"` for `D(y | x)`), `DescTools::KendallTauB`.
- PySpark: N/A — once joint counts are aggregated (via `techniques/chi-square-tests/pyspark/`), the rest is on the driver.

## Run
```
python techniques/goodman-kruskal-somers/python/goodman_kruskal_somers.py
Rscript techniques/goodman-kruskal-somers/r/goodman_kruskal_somers.R
```

**Refs:** Goodman & Kruskal, "Measures of Association for Cross Classifications," *JASA* 49(268), 732–764, 1954; Somers, "A New Asymmetric Measure of Association for Ordinal Variables," *American Sociological Review* 27(6), 799–811, 1962.

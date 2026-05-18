# Effect Size Measures (Reference §1.6, §1.25)

Effect sizes describe the **magnitude** of a difference or association — independent of sample size. Always report them, ideally with confidence intervals, alongside p-values.

## Standardized mean differences (two groups)
| Measure | Definition | When |
|---------|-----------|------|
| Cohen's d | `(x̄₁ − x̄₂) / s_pooled` | Standard SMD; pooled SD assumes ≈ equal variances |
| Hedges' g | `d · J(df)`, `J = 1 − 3/(4·df − 1)` | Cohen's d with small-sample bias correction |
| Glass's Δ | `(x̄_treat − x̄_ctrl) / s_control` | When group variances differ a lot (use the control SD) |

## Nonparametric (ordinal / ranks)
| Measure | Definition | Notes |
|---------|-----------|-------|
| Cliff's delta | `P(X₁>X₂) − P(X₁<X₂)` ∈ [−1,1] | Dominance; robust to outliers/non-normality |
| Rank-biserial r | `2·U₁/(n₁n₂) − 1` from Mann–Whitney U₁ | Algebraically equals Cliff's delta |

## Variance explained (k groups / ANOVA)
| Measure | Definition | Notes |
|---------|-----------|-------|
| η² (eta-squared) | `SS_between / SS_total` | Proportion of variance explained; upward-biased |
| ω² (omega-squared) | `(SS_b − (k−1)·MS_w) / (SS_total + MS_w)` | Less biased estimate of the population value |
| Cohen's f | `√(η² / (1 − η²))` | Used in power analysis for ANOVA |

## Conventional benchmarks (Cohen 1988 — context always wins)
`d`: 0.2 / 0.5 / 0.8 · `r`: 0.1 / 0.3 / 0.5 · `η²`: 0.01 / 0.06 / 0.14 · `f`: 0.1 / 0.25 / 0.4 (small / medium / large). A `d = 0.2` can be life-or-death for a mortality outcome and trivial for satisfaction scores — prefer field-specific benchmarks (Funder & Ozer 2019).

## Files
- `python/effect_sizes.py` — from-scratch d/g/Δ, Cliff's delta, rank-biserial, η²/ω²/f, plus an `interpret()` helper; compares against `pingouin.compute_effsize` and a `scipy.stats.mannwhitneyu`-derived rank-biserial
- `r/effect_sizes.R` — from-scratch + `effsize` (`cohen.d`, `cliff.delta`), `effectsize` (`hedges_g`, `eta_squared`)
- PySpark: N/A (effect sizes summarize a study, not a big-data computation)

## Run
```
python techniques/effect-sizes/python/effect_sizes.py
Rscript techniques/effect-sizes/r/effect_sizes.R
```

**Refs:** Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., 1988; Fritz, Morris & Richler, "Effect Size Estimates," *J. Exp. Psych: General* 141(1), 2–18, 2012; Funder & Ozer, "Evaluating Effect Size in Psychological Research," *AMPPS* 2(2), 156–168, 2019.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)

# Eta Correlation Ratio (Reference §4.13)

The "correlation" between a **categorical** grouping variable `X` (k levels) and a **continuous** variable `Y`. Defined as the proportion of variance in `Y` explained by group membership:

`η² = SS_between / SS_total`,  `η = √η²` ∈ `[0, 1]`.

## Connections to things you already know

- **One-way ANOVA**: `η²` is exactly the model R² for the one-way ANOVA of `Y` on `X` (treating `X` as a factor). The F-test for `η² = 0` is the same F-test as in ANOVA.
- **Point-biserial**: for `k = 2` groups, `η = |r_pb|` (`techniques/point-biserial-correlation`).
- **Multiple correlation**: `η = R` from regressing `Y` on the indicator-coded `X`.
- **Effect-size repository**: `η²` also appears in `techniques/effect-sizes` as the ANOVA effect size (§1.6/§1.25). Same number, different framing — here it's a *correlation*, there it's an *effect size*.

## What it captures (and what it doesn't)

`η²` measures **any** difference in conditional means of `Y` across the categories of `X`. It does **not** assume linearity (the categories aren't even ordered). For two genuinely *ordinal* variables, use Spearman or Kendall instead.

`η²` can be much larger than the linear `r²` you'd get from coding `X` as `1, 2, …, k`: if the group means are non-monotone, the linear coding misses the relationship but `η²` doesn't.

## Significance

`F = (η² / (k − 1)) / ((1 − η²) / (N − k))` ~ `F(k − 1, N − k)` — identical to the one-way ANOVA F-test.

## Why list this separately from `effect-sizes`

`effect-sizes` frames `η²` as "the effect size accompanying a one-way ANOVA"; this folder frames it as "a correlation between a categorical and a continuous variable" (the framing the reference doc uses in Ch. 4). The math is identical; the directory exists so it's findable both ways.

## Files
- `python/eta_correlation_ratio.py` — from-scratch `η`/`η²` from `SS_between / SS_total` + the F-test; works either from a list of per-group y-values or from parallel `(x, y)` columns; compares against `pingouin.anova` when available.
- `r/eta_correlation_ratio.R` — from-scratch + base `aov` and `effectsize::eta_squared`.
- PySpark: N/A — reuse the per-group sufficient statistics from `techniques/one-way-anova/pyspark/` and compute `η²` from `SS_between` / `SS_total` on the driver.

## Run
```
python techniques/eta-correlation-ratio/python/eta_correlation_ratio.py
Rscript techniques/eta-correlation-ratio/r/eta_correlation_ratio.R
```

**Ref:** Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., 1988.

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

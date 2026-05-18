# Categorical Variable Coding Schemes (Reference §5.32)

A categorical predictor with `k` levels needs `k − 1` columns in the design matrix. The four standard schemes all give **identical fitted values, R², and σ̂** — they differ only in what the **individual coefficients** mean.

## Schemes

| Scheme | Coding | What `β_j` means | Intercept means |
|--------|--------|------------------|------------------|
| **Dummy** (Treatment) | `1` if level `j`, else `0` (reference level omitted) | `mean(level_j) − mean(reference)` | `mean(reference)` |
| **Effect** (Sum-to-zero) | `+1` if level `j`, `−1` if reference, else `0` | `mean(level_j) − grand_mean` | grand mean |
| **Helmert** | Successive contrasts | `mean(level_{j+1}) − mean(levels_1..j)` (the *added* level vs. all prior) | mean of level 1 (or grand, depending on scaling) |
| **Deviation** | `+1` if level `j`, `−1` if the **last** level, else `0` | `mean(level_j) − grand_mean` | grand mean |

`effect` and `deviation` differ only in **which** level is omitted (first vs. last); the substantive parameterization is the same family.

## When to pick which

- **Dummy** — comparing groups to a meaningful **reference** (e.g. control). The default in R, pandas, statsmodels.
- **Effect** — comparing each level to the **overall average**. The natural parameterization for ANOVA-style "main effects + interactions" tables.
- **Helmert** — **ordered** categorical (low / medium / high) with theory-driven sequencing: "does adding level 2 differ from level 1?" then "does adding level 3 differ from the average of 1, 2?" etc.
- **Deviation** — same use as effect, just with the convention of omitting the last level.

## "Why did my coefficient flip sign?"

The most common cause of this question: the analyst changed the reference level (often by sorting differently or refactoring) and now the coefficients are differences against a different baseline. Three of the schemes don't even produce a `"vs reference"` coefficient at all — they produce `"vs grand mean"` or `"vs prior levels"`. Read what your software is doing before reading the sign.

## R vs. Python defaults

- **R** uses `contr.treatment` (dummy) by default for unordered factors. Change globally with `options(contrasts = c("contr.sum", "contr.poly"))`.
- **statsmodels** (Python) defaults to dummy (`Treatment`) when you write `C(g)` in a formula; specify `C(g, Sum)` / `C(g, Helmert)` for the others.
- **scikit-learn**'s `OneHotEncoder(drop="first")` is dummy coding; without `drop`, it's a redundant full-rank-deficient encoding only suitable for tree models.

## Files
- `python/categorical_variable_coding.py` — from-scratch builders for all four schemes, plus a one-factor ANOVA fit that prints the coefficient table per scheme. Demos that the fitted values are byte-identical across schemes (max abs difference < 10⁻¹⁴). Cross-checks dummy / effect against `statsmodels` `C(g, Treatment)` / `C(g, Sum)`.
- `r/categorical_variable_coding.R` — from-scratch + base `contr.treatment`, `contr.sum`, `contr.helmert` via the `C(factor, contrast)` formula syntax.
- PySpark: N/A — Spark's `StringIndexer` + `OneHotEncoder` produces dummy coding; for other schemes build the columns explicitly.

## Run
```
python techniques/categorical-variable-coding/python/categorical_variable_coding.py
Rscript techniques/categorical-variable-coding/r/categorical_variable_coding.R
```

**Refs:** Cohen, Cohen, West & Aiken, *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*, 3rd ed., 2003 (Ch. 8); Faraway, *Linear Models with R*, 2nd ed., 2014 (§14).

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

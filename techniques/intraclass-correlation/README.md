# Intraclass Correlation Coefficient (Reference §4.6)

Quantifies the share of total variance attributable to **between-subject** differences rather than **within-subject** (measurement) noise. The standard reliability statistic for `n` subjects each rated by `k` raters.

## Shrout & Fleiss (1979) / McGraw & Wong (1996) classification

Six common variants depending on the design and what unit you report:

| Variant | Model | Form | Unit | Use |
|---------|-------|------|------|-----|
| `ICC(1, 1)` | One-way ANOVA (raters random, possibly different per subject) | absolute | single | Each subject rated by *some* raters drawn from a population |
| `ICC(1, k)` | one-way | absolute | average of `k` | … and you report the mean of `k` ratings |
| `ICC(A, 1)` / `ICC(2, 1)` | Two-way random (`k` raters are a sample) | **absolute** agreement | single | Raters could differ in level; you want agreement in raw value |
| `ICC(A, k)` / `ICC(2, k)` | two-way random | absolute | average | Reliability of the **mean** of `k` ratings |
| `ICC(C, 1)` / `ICC(3, 1)` | Two-way mixed (`k` raters are the only raters of interest) | **consistency** | single | Only the relative ordering across subjects matters |
| `ICC(C, k)` / `ICC(3, k)` | two-way mixed | consistency | average | … averaged across the fixed `k` raters |

`absolute` agreement penalizes between-rater bias; `consistency` ignores it (a rater who's always +5 from everyone else still has perfect consistency).

## Computation

Run a two-way ANOVA (subjects × raters, no interaction). Get `MS_R` (rows), `MS_C` (columns), `MS_E` (residual). Then with `n` subjects, `k` raters:

- `ICC(A, 1) = (MS_R − MS_E) / (MS_R + (k−1)·MS_E + k(MS_C − MS_E)/n)`
- `ICC(A, k) = (MS_R − MS_E) / (MS_R + (MS_C − MS_E)/n)`
- `ICC(C, 1) = (MS_R − MS_E) / (MS_R + (k−1)·MS_E)`
- `ICC(C, k) = (MS_R − MS_E) / MS_R`
- `ICC(1, 1) = (MS_B − MS_W) / (MS_B + (k−1)·MS_W)` (one-way: only between/within)

## Conventional benchmarks (Koo & Li 2016)

| ICC | Reliability |
|-----|-------------|
| < 0.50 | Poor |
| 0.50 – 0.75 | Moderate |
| 0.75 – 0.90 | Good |
| > 0.90 | Excellent |

Field-dependent — clinical-measurement work often demands > 0.90 for individual decisions, > 0.75 for group-level research.

## Files
- `python/intraclass_correlation.py` — from-scratch two-way ANOVA `MS_R/MS_C/MS_E` plus all six variants; demos all six on a single synthetic dataset where rater bias is present (so `(A,*)` and `(C,*)` differ); compares against `pingouin.intraclass_corr` when available.
- `r/intraclass_correlation.R` — from-scratch + `psych::ICC` (which prints all six in one shot) and notes on `irr::icc`.
- PySpark: N/A — ICC fits a small subjects-by-raters table.

## Run
```
python techniques/intraclass-correlation/python/intraclass_correlation.py
Rscript techniques/intraclass-correlation/r/intraclass_correlation.R
```

**Refs:** Shrout & Fleiss, "Intraclass Correlations: Uses in Assessing Rater Reliability," *Psychological Bulletin* 86(2), 420–428, 1979; McGraw & Wong, "Forming Inferences About Some Intraclass Correlation Coefficients," *Psychological Methods* 1(1), 30–46, 1996; Koo & Li, "A Guideline of Selecting and Reporting Intraclass Correlation Coefficients for Reliability Research," *J. Chiropr. Med.* 15(2), 155–163, 2016.

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

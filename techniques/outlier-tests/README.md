# Outlier Tests (Reference §3.25)

Formal tests of "is this extreme value farther from the mean than we'd expect under a normal distribution?" — together with one widely used non-test heuristic (Tukey's IQR rule).

## Tests

| Test | Detects | Sample size |
|------|---------|-------------|
| **Grubbs** | A single most-extreme value (one-sided or two-sided) | `n ≥ ~6` |
| **Generalized ESD** (Rosner) | Up to `r` outliers (iteratively masks the most extreme and re-tests) | `n ≥ 10`, ideal |
| **Dixon's Q** | A single outlier via gap/range; uses tabulated critical values | `3 ≤ n ≤ ~30` |
| **Tukey IQR rule** | Points outside `[Q1 − k·IQR, Q3 + k·IQR]` (`k=1.5` outlier, `k=3` "far out"). **Not** a hypothesis test — just a robust description. | any |

## Important caveats

- **"Outlier" ≠ "error".** Before deleting flagged points, *investigate*. The extreme value may be the most informative observation in the dataset.
- **These tests assume the rest of the data are normal.** On skewed or heavy-tailed data they over-flag — they'll declare a point in the long tail "an outlier" when it's perfectly typical for that distribution. Check normality first (`techniques/normality-tests`) or transform the data.
- **Better defaults than test-and-delete**: a robust statistical method (`techniques/robust-location-scale` — Yuen's trimmed t, Huber M-estimators, MAD) keeps every observation and just down-weights extreme ones.
- **Repeated retesting changes things.** After Grubbs flags and removes a point, running Grubbs again on the remaining data inflates the type-I error. Use Generalized ESD if you really expect multiple outliers.

## Tukey's IQR rule

The non-test version most people reach for: `lower = Q1 − 1.5·IQR`, `upper = Q3 + 1.5·IQR`. Robust, makes no normality assumption, doesn't pretend to be a hypothesis test. It's what's behind the "whiskers" in a standard boxplot (`Reference §1.10`, `§1.31`).

## Files
- `python/outlier_tests.py` — from-scratch Grubbs / Generalized ESD / Dixon's Q (with the standard tabulated critical values) / IQR rule. No library equivalent in stock scipy.
- `r/outlier_tests.R` — from-scratch + `outliers::grubbs.test`, `outliers::dixon.test`, `EnvStats::rosnerTest` for Generalized ESD.
- PySpark: N/A — these are sample-level tests. For large data, compute `Q1` / `Q3` via `approxQuantile` (`techniques/quantiles/pyspark/`) and apply the IQR rule as a filter.

## Run
```
python techniques/outlier-tests/python/outlier_tests.py
Rscript techniques/outlier-tests/r/outlier_tests.R
```

**Refs:** Grubbs, "Procedures for Detecting Outlying Observations in Samples," *Technometrics* 11(1), 1–21, 1969; Rosner, "Percentage Points for a Generalized ESD Many-Outlier Procedure," *Technometrics* 25(2), 165–172, 1983; Dixon, "Analysis of Extreme Values," *Ann. Math. Stat.* 21(4), 488–506, 1950; Tukey, *Exploratory Data Analysis*, Addison-Wesley, 1977.

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

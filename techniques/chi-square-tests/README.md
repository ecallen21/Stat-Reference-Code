# Chi-Square Tests (Reference §3.5)

Pearson's `X² = Σ (O − E)² / E` applied to two classical problems on count data.

## Goodness-of-fit (GOF)
Does the observed distribution of category counts match a hypothesized distribution?
- `E_i = n · p_i^null`, where `p_i^null` is the null probability of category `i`.
- `X² = Σ (O_i − E_i)² / E_i`, `df = k − 1 − (# parameters estimated)`.
- Example: is a die fair? `p_null = (1/6, …, 1/6)`.

## Test of independence (r × c contingency table)
Are the row and column variables independent?
- Under independence: `E_ij = (row_total_i × col_total_j) / n`.
- `X² = Σ_ij (O_ij − E_ij)² / E_ij`, `df = (r − 1)(c − 1)`.

## Yates' continuity correction (2×2 only)
For a 2×2 table with small `n`, replace `(O − E)²` with `(|O − E| − 0.5)²` (floored at 0). Reduces over-rejection. R's `chisq.test` applies it by default for 2×2; this implementation makes it explicit.

## Effect size
**Cramer's V** = `√(X² / (n · min(r−1, c−1)))` ∈ [0, 1]. For 2×2 it equals the phi coefficient. Conventional benchmarks (small/medium/large): df=1 → 0.10/0.30/0.50; df=2 → 0.07/0.21/0.35; df=3 → 0.06/0.17/0.29.

## Pearson standardized residuals
`r_ij = (O_ij − E_ij) / √E_ij`. After a significant overall test, `|r_ij| > 2` points to the cells driving the result — like post-hoc cell inspection.

## Assumptions / when not to use it
The X² distribution is an **approximation**; it needs the expected counts to be reasonably large.
- Rule of thumb: `E_ij ≥ 5` in most cells (Cochran).
- If many cells fail that — use **Fisher's exact test** (`techniques/fisher-exact`) or a Monte-Carlo p-value (`chisq.test(..., simulate.p.value = TRUE)`).
- Independence assumes one observation per subject. For matched/paired binary data use McNemar's test (Ch. 8, future batch).

## Files
- `python/chi_square_tests.py` — from-scratch GOF + independence (with Yates option) + Cramer's V + standardized residuals; compares against `scipy.stats.chi2_contingency` / `chisquare`.
- `r/chi_square_tests.R` — from-scratch versions + `stats::chisq.test` (which gives observed, expected, residuals, and standardized residuals via `$stdres`).
- `pyspark/chi_square_tests.py` — `groupBy(row_col, col_col).count()` aggregates the joint counts on the cluster; the small contingency table is collected and the chi-square is computed on the driver. (Spark MLlib also exposes `ChiSquareTest` for the same purpose.)

## Run
```
python techniques/chi-square-tests/python/chi_square_tests.py
Rscript techniques/chi-square-tests/r/chi_square_tests.R
python techniques/chi-square-tests/pyspark/chi_square_tests.py
```

**Refs:** Agresti, *Categorical Data Analysis*, 3rd ed., Wiley, 2013; Cochran, "The χ² Test of Goodness of Fit," *Ann. Math. Stat.* 23(3), 315–345, 1952.

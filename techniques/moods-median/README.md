# Mood's Median Test (Reference §6.8)

A simple, robust test that the **medians** of `k` groups are equal.

## Algorithm

1. Pool everything; compute the overall median `M`.
2. Build a `2 × k` contingency table: row 1 = "# values > M", row 2 = "# values ≤ M", one column per group.
3. Chi-square test of independence on the table (or Fisher's exact for small `n`).

## Vs. Kruskal–Wallis

- **K–W** uses **all rank information**; more powerful when group shapes are similar.
- **Mood's median** only tracks "above or below the pooled median" — throws away magnitudes.
- When the groups have **different shapes / scales**, K–W can be significant for shape reasons; Mood's is purely about location and is **more interpretable as a median test** in that situation.

Rule of thumb: report K–W first; if you specifically need a "do the medians differ?" claim independent of shape, add Mood's median.

## Files
- `python/moods_median.py` — pooled median, contingency table, chi-square; matches `scipy.stats.median_test` exactly.
- `r/moods_median.R` — from-scratch via `chisq.test`; notes on `RVAideMemoire::mood.medtest`.
- PySpark: N/A.

## Run
```
python techniques/moods-median/python/moods_median.py
Rscript techniques/moods-median/r/moods_median.R
```

**Ref:** Mood, *Introduction to the Theory of Statistics*, McGraw-Hill, 1950.

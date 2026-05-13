# Fisher's Exact Test (Reference §3.6)

For a 2×2 contingency table

```
           col 1   col 2   row totals
   row 1    a       b        a + b
   row 2    c       d        c + d
   totals   a+c    b+d     n = a+b+c+d
```

Fisher's exact test conditions on **both** sets of marginal totals and computes the *exact* probability of seeing a table at least as extreme as observed under H₀ (independence / OR = 1), via the hypergeometric distribution:

`P(X = a | margins) = C(a+b, a) · C(c+d, c) / C(n, a+c)`

The two-sided p-value (this implementation, and scipy) is the sum of all hypergeometric probabilities `≤ p_observed`. R's `fisher.test` uses the same definition for `2×2`.

## When to use it (instead of chi-square)
- Any expected count `< ~5` (Cochran's rule of thumb).
- Very small total `n`.
- Rare events / sparse 2×2.

Chi-square is asymptotic; Fisher's is exact. For large tables chi-square is faster and effectively identical.

## Effect size: the odds ratio
`OR = (a·d) / (b·c)`. `OR = 1` ↔ independence. The Wald CI on `log(OR)` uses
`SE(log OR) = √(1/a + 1/b + 1/c + 1/d)` (with the Haldane–Anscombe `+0.5` correction if any cell is 0). R's `fisher.test` returns a tighter **conditional exact** CI — both are valid; reach for the exact one when cells are tiny.

## Generalizations
- **r × c exact test (Fisher–Freeman–Halton)**: `scipy.stats.fisher_exact` (1.11+) and R's `fisher.test` handle larger tables exactly.
- **McNemar's test** (Ch. 8 future batch): for paired binary outcomes, not independent samples.
- **Mid-p variant**: subtracts half the probability of the observed table from the p-value; less conservative, similar to Lancaster's mid-p for the binomial.

## Files
- `python/fisher_exact.py` — from-scratch hypergeometric p-value (two-sided / greater / less) + OR + Wald log-OR CI (with zero-cell correction); compares against `scipy.stats.fisher_exact`.
- `r/fisher_exact.R` — from-scratch + `stats::fisher.test` (which provides the conditional exact CI on the OR).
- PySpark: N/A — Fisher's exact is for *small* tables. If your data are huge enough to need Spark, the chi-square approximation in `techniques/chi-square-tests/pyspark/` is appropriate.

## Run
```
python techniques/fisher-exact/python/fisher_exact.py
Rscript techniques/fisher-exact/r/fisher_exact.R
```

**Refs:** Fisher, *Statistical Methods for Research Workers*, Oliver & Boyd, 1925; Mehta & Patel, "A Network Algorithm for Performing Fisher's Exact Test in r × c Contingency Tables," *JASA* 78(382), 1983.

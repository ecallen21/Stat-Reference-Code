# Kendall's Tau (Reference §4.3)

A rank correlation defined **directly from pairs of observations**, rather than via ranks. For every pair `(i, j)`:

- **Concordant**: `(xᵢ − xⱼ)` and `(yᵢ − yⱼ)` have the **same sign**.
- **Discordant**: opposite signs.
- **Tied**: either delta is zero (in `x` only, in `y` only, or in both).

Let `C`, `D` = counts of concordant / discordant pairs, `Tₓ`, `T_y`, `T_xy` = the tie counts, and `n₀ = n(n−1)/2`. Then:

| Variant | Formula | Use |
|---------|---------|-----|
| **τ-a** | `(C − D) / n₀` | No ties only |
| **τ-b** | `(C − D) / √((n₀ − Tₓ)(n₀ − T_y))` | **Default** — handles ties; matches `scipy` and R |
| τ-c (not implemented) | rectangular tables of different `r × c` | Asymmetric ordinal tables |

## Significance

`z = (C − D) / √(n(n−1)(2n+5)/18)` is asymptotically standard normal under H₀ (no association). This is the form scipy reports; an exact small-sample distribution exists but quickly becomes infeasible.

## Interpretation

`τ` has a clean interpretation: `τ = P(concordant) − P(discordant)`. So `τ = 0.5` means a randomly drawn pair is 50 percentage points more likely to be concordant than discordant.

## Tau vs. Spearman vs. Pearson

- **Pearson** — linearity assumed; sensitive to outliers; CI via Fisher z.
- **Spearman** — monotone, not linearity; robust to outliers; computed via ranks.
- **Kendall** — monotone, robust, and the most interpretable (`P(concordant) − P(discordant)`); slower to compute (`O(n²)` naively).

For most ordinal-data work, Kendall is the more defensible choice; Spearman is often reported because it's faster historically. Both are within 0.05–0.10 of each other on well-behaved data.

## Files
- `python/kendalls_tau.py` — from-scratch `O(n²)` pair-counting, τ-a, τ-b, and the z-test; compares against `scipy.stats.kendalltau(variant="b")`. Includes an ordinal-with-ties example.
- `r/kendalls_tau.R` — from-scratch versions + `stats::cor(method = "kendall")` / `stats::cor.test(method = "kendall")`.
- `pyspark/kendalls_tau.py` — `O(n²)` self-join on `id < id` then aggregate concordant/discordant/tied counts. Fine for moderate `n`; for very large data, consider sampling first (Spark's `Correlation.corr` doesn't ship Kendall — Pearson and Spearman only).

## Run
```
python techniques/kendalls-tau/python/kendalls_tau.py
Rscript techniques/kendalls-tau/r/kendalls_tau.R
python techniques/kendalls-tau/pyspark/kendalls_tau.py
```

**Refs:** Kendall, "A New Measure of Rank Correlation," *Biometrika* 30(1/2), 81–93, 1938; Knight, "A Computer Method for Calculating Kendall's Tau with Ungrouped Data," *JASA* 61(314), 436–439, 1966 (the O(n log n) algorithm).

# Sign Test (Reference §6.1)

The minimal-assumption test about a **median** (or paired differences). Uses only the *signs* of the observations relative to the null value — no magnitudes, no symmetry, no normality.

## One-sample form

`H₀: median(X) = m₀`. Let `S = #{xᵢ > m₀}`; drop ties.

Under H₀, `S ~ Binomial(n_no_tie, 0.5)` — exactly. The p-value is computed from the binomial distribution (the same engine as `techniques/binomial-test`).

## Paired form

Test `H₀: median(X₁ − X₂) = 0` by applying the one-sample sign test to the differences. Same as `binom.test(#{positive}, #{non-zero}, 0.5)`.

## Vs. Wilcoxon signed-rank (`techniques/wilcoxon-signed-rank`)

| Test | Information used | Symmetry assumption |
|------|------------------|----------------------|
| Sign | Sign of `(xᵢ − m₀)` only | None |
| Wilcoxon signed-rank | Sign **and** rank of `\|xᵢ − m₀\|` | Yes (symmetric around the median under H₀) |

The Wilcoxon is more powerful when symmetric, but the sign test is **always valid** — even for badly skewed differences where Wilcoxon's symmetry assumption breaks. For paired differences with clear skew, prefer the sign test.

## Caveats

- **Low power** — using only signs throws away information. Need a meaningful effect for the test to find it at typical sample sizes.
- **Ties on `m₀`** are dropped, so a flat distribution loses observations fast.
- For tiny `n` (e.g. 5), the test is intrinsically conservative — discreteness of the binomial limits the p-values.

## Files
- `python/sign_test.py` — one-sample and paired sign tests via `scipy.stats.binomtest`; compares against `statsmodels.stats.descriptivestats.sign_test`.
- `r/sign_test.R` — from-scratch via `binom.test`; notes on `BSDA::SIGN.test`.
- PySpark: N/A — `binom.test` from cluster-aggregated `#{positive}` and `#{non-zero}` counts is trivial.

## Run
```
python techniques/sign-test/python/sign_test.py
Rscript techniques/sign-test/r/sign_test.R
```

**Refs:** Conover, *Practical Nonparametric Statistics*, 3rd ed., Wiley, 1999; Hollander, Wolfe & Chicken, *Nonparametric Statistical Methods*, 3rd ed., Wiley, 2014.

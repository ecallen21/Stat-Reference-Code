# Tests for Normality (Reference §3.19, §3.40)

Formal tests of `H₀ : the data come from a normal distribution`. Use these to *check* an assumption made by a downstream procedure (t-test, ANOVA, OLS residuals), but read the caveats below first — a Q-Q plot is usually more informative than a single p-value.

## Tests covered

| Test | Statistic | Strengths |
|------|-----------|-----------|
| **Shapiro–Wilk** | ratio of two variance estimators based on order statistics | The default; most powerful general-purpose test. `n ≤ 5000` in R. |
| **Jarque–Bera** | `(n/6)(g₁² + g₂²/4)` ~ χ²₂; from sample skewness `g₁` and excess kurtosis `g₂` | Easy to compute; asymptotic only — poor for `n < ~100`. |
| **D'Agostino–Pearson K²** | transforms `g₁`, `g₂` to standard normals; omnibus on skew + kurt | Good for `n ∈ [20, 1000+]`; widely used. |
| **Lilliefors** | Kolmogorov–Smirnov distance against `N(x̄, s²)` with *estimated* parameters | Don't use plain K–S with fitted parameters — Lilliefors gives the correct critical values. |
| **Anderson–Darling** | weighted CDF distance, extra weight in the tails | Sensitive to tail deviations specifically. |

## What the p-value tells you (and doesn't)

- **Failure to reject ≠ normal.** With small `n`, every test is underpowered — you can't conclude normality from a non-significant result.
- **Rejection ≠ unusable.** With huge `n`, every test rejects, because *no* real-world data are exactly normal. A tiny deviation passes statistical significance long before it has any practical effect on a t-test or OLS.
- **Always pair with a Q-Q plot.** §1.29 of the reference doc covers Q-Q interpretation; the eye sees patterns (S-shape, fat tails, ties) that a single p-value can't.
- For OLS / ANOVA, the assumption is normality of the **residuals**, not the raw outcome. Run the test on `resid(fit)`, not on `y`.

## A practical workflow

1. Plot the data: histogram + density, then Q-Q vs. normal (Reference §1.9, §1.29).
2. Compute skewness / kurtosis (`techniques/shape-skewness-kurtosis`).
3. Run **one** test as a tiebreaker — Shapiro–Wilk is the default unless `n` is very large.
4. If badly non-normal:
   - try a transformation (log for right-skewed positive data; `techniques/coefficient-of-variation` notes the geometric mean view);
   - or use a nonparametric / robust alternative (Wilcoxon / Mann–Whitney, Yuen's trimmed t in `techniques/robust-location-scale`, bootstrap).

## Files
- `python/normality_tests.py` — from-scratch Jarque–Bera and a Monte-Carlo-p Lilliefors (so we don't need tabulated critical values); Shapiro–Wilk / D'Agostino–Pearson / Anderson–Darling via scipy.
- `r/normality_tests.R` — from-scratch versions + `stats::shapiro.test`, `nortest::lillie.test`, `nortest::ad.test`, `tseries::jarque.bera.test`.
- PySpark: N/A — these are small-sample assumption checks. (At Spark scale you'll be rejecting normality on any non-degenerate sample; use a Q-Q plot of a random sub-sample instead.)

## Run
```
python techniques/normality-tests/python/normality_tests.py
Rscript techniques/normality-tests/r/normality_tests.R
```

**Refs:** Shapiro & Wilk, "An Analysis of Variance Test for Normality (Complete Samples)," *Biometrika* 52, 591–611, 1965; D'Agostino & Pearson, "Tests for Departure from Normality," *Biometrika* 60, 613–622, 1973; Lilliefors, "On the Kolmogorov–Smirnov Test for Normality with Mean and Variance Unknown," *JASA* 62, 399–402, 1967.

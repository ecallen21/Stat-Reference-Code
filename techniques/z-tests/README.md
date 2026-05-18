# z-tests (Reference §3.7, §3.22)

Large-sample / known-variance cousins of the t-test:
`z = (estimate − null) / SE_under_null`, compared to the standard normal.

| Variant | H₀ | SE under H₀ |
|---------|----|-------------|
| **One-sample mean** (σ known) | `mean(x) = μ₀` | `σ / √n` |
| **Two-sample mean** (σ₁, σ₂ known) | `mean(x₁) = mean(x₂)` | `√(σ₁²/n₁ + σ₂²/n₂)` |
| **One proportion** | `p = p₀` | `√(p₀(1−p₀)/n)` |
| **Two proportions** | `p₁ = p₂` | `√(p̂_pool(1−p̂_pool)(1/n₁ + 1/n₂))`, `p̂_pool = (x₁+x₂)/(n₁+n₂)` |

**Continuity correction (Yates)** — for small `n`, subtract `1/(2n)` (one sample) or `½·(1/n₁ + 1/n₂)` (two samples) from `|p̂ − p₀|` before forming z. R's `prop.test` applies it by default; this implementation makes it an opt-in.

## When to use z vs. t

- **Means**: in practice, σ is almost never known — use a t-test. The mean z-test is mostly a teaching tool / a large-`n` approximation.
- **Proportions**: counts are integers and there is no "sample SD" to estimate; the z (or its squared form, the chi-square test) is the standard.
- **A/B testing**: two-proportion z is the workhorse. For very small `n` or rare events, prefer Fisher's exact test (`techniques/fisher-exact`).

## SE under the null vs. for the CI

The **test** uses the SE *under H₀* (so it pools `p̂_pool` for the two-proportion z). The **confidence interval** on the difference uses the *unpooled* Wald SE `√(p̂₁(1−p̂₁)/n₁ + p̂₂(1−p̂₂)/n₂)` — because under H₁ the proportions are different. This is why the test can be significant but the Wald CI on the difference can include 0 if you mix the two SEs; both are reported here.

For the one-proportion CI itself, prefer Wilson / Clopper–Pearson (see `techniques/rates-proportions`); the Wald CI is shown here for symmetry with the test.

## Files
- `python/z_tests.py` — from-scratch one/two-sample mean z and one/two-proportion z (with optional Yates correction); compares against `statsmodels.stats.proportion.proportions_ztest` and `proportion_confint`.
- `r/z_tests.R` — from-scratch versions + `stats::prop.test` (the chi-square / z² version with continuity correction).
- `pyspark/z_tests.py` — distributed two-proportion test: per-group counts via `groupBy → sum/count`, then the closed-form pooled-SE z on the driver. The standard pattern for A/B tests on huge datasets.

## Run
```
python techniques/z-tests/python/z_tests.py
Rscript techniques/z-tests/r/z_tests.R
python techniques/z-tests/pyspark/z_tests.py
```

**Refs:** Agresti & Coull, "Approximate is Better than 'Exact' for Interval Estimation of Binomial Proportions," *The American Statistician* 52(2), 119–126, 1998; Newcombe, "Interval Estimation for the Difference Between Independent Proportions," *Statistics in Medicine* 17, 873–890, 1998.

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

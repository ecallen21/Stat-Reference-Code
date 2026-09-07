# Cochran-Armitage Trend Test (Reference §4.16)

Cochran (1954); Armitage (1955). Tests for a **linear trend** in a
binary outcome across `K` ordered categories (e.g., dose levels or
exposure tertiles). More powerful than the omnibus chi-square when
the alternative is truly monotone.

## Statistic

Groups `i = 1, …, K` with `n_i` subjects and `r_i` events, scored
by `x_i`:

    T = Σᵢ x_i · (r_i − n_i · p̄) /
        √( p̄ (1 − p̄) · [Σᵢ n_i x_i² − N x̄²] )

with `p̄ = Σ r_i / N`, `x̄ = Σ n_i x_i / N`. Under `H₀` (no trend)
`T ~ N(0, 1)`.

## Files

- `python/cochran_armitage_trend.py` — from-scratch CAT + omnibus
  chi-square comparison. Demo (K=4, monotone dose response
  0.10 → 0.40): CAT Z=+5.36 p≈9e-8, chi-square p≈1e-6. Non-monotone
  spike (0.15, 0.40, 0.20, 0.15): CAT Z=-1.07 p=0.28 (no linear
  signal) but chi-square p≈2e-5 (rejects independence).
- `r/cochran_armitage_trend.R` — `stats::prop.trend.test`,
  `DescTools::CochranArmitageTest`, `coin::independence_test`
  (R); from-scratch (Python).

## When to use

- **Dose-response** — biomarker across dose tertiles, exposure
  quintiles, ordinal SES bins.
- **Ordinal predictor, binary outcome** — screening for monotone
  associations.
- **Higher power than chi-square** — CAT concentrates power on the
  linear direction.

## When NOT to use

- **Non-monotone relationships** — CAT can miss U-shaped or
  threshold effects; use omnibus chi-square or a shape-flexible
  test.
- **Unordered categories** — no natural ordering.
- **Categorical predictor, categorical (non-binary) outcome** — use
  ordinal-response models.

## Assumptions & caveats

- **Scores `x_i`** — usually equal-spaced (1, 2, …, K) but can be
  actual dose values; the test statistic depends on the choice.
- **Independence within cells** — assumes iid subjects; use design-
  based extensions for clustered data.
- **Large-sample normal approximation** — for small cells use the
  exact permutation version (coin::independence_test).
- **Continuity correction** — some software applies Yates-style
  correction; typically unimportant with n ≥ 100 per cell.

## Related in this repo

- `chi-square-tests`, `fisher-exact`, `cochran-mantel-haenszel` —
  categorical-testing cousins.
- `logistic-regression`, `ordinal-logistic` — model-based
  alternatives.
- `jonckheere-terpstra` — nonparametric ordered-groups test.
- `runs-test`, `mann-whitney` — nonparametric trend cousins.

## Run

```
python techniques/cochran-armitage-trend/python/cochran_armitage_trend.py
Rscript techniques/cochran-armitage-trend/r/cochran_armitage_trend.R
```

**Refs:** Cochran, W.G. "Some methods for strengthening the common chi-squared tests." *Biometrics*, 10(4): 417-451, 1954; Armitage, P. "Tests for linear trends in proportions and frequencies." *Biometrics*, 11(3): 375-386, 1955.

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

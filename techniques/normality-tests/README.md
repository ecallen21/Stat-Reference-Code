# Normality Tests (Reference §3.24, §3.25)

Shapiro-Wilk (1965), Stephens (1974), Jarque-Bera (1980). Four
classical omnibus tests of `H₀: y ~ Normal`:

| Test | Idea |
|---|---|
| **Shapiro-Wilk** | Correlation between order statistics and normal quantiles |
| **Anderson-Darling** | Weighted Cramér-von Mises with tail emphasis |
| **Jarque-Bera** | Moments (skew + excess kurtosis) |
| **Kolmogorov-Smirnov** | Max abs distance between ECDF and N(0,1) |

Choose:
- Small n (< 50) → Shapiro-Wilk.
- Moderate n → Shapiro or Anderson-Darling.
- Very large n (> 5000) → tests almost always reject; use QQ-plot
  and effect-size judgement instead.

## Files

- `python/normality_tests.py` — all four via `scipy.stats`. Demo
  (n=200): Normal(0,1) → all four fail-to-reject; Uniform(0,1)
  → Shapiro, AD, JB reject strongly; Exponential → all reject;
  t(3) → all reject.
- `r/normality_tests.R` — `stats::shapiro.test`, `nortest::ad.test`
  / `lillie.test` / `cvm.test`, `tseries::jarque.bera.test` (R);
  `scipy.stats.shapiro`/`anderson`/`jarque_bera`/`kstest`,
  `statsmodels.stats.diagnostic.lilliefors` (Python).

## When to use

- **Diagnostic step** before applying normal-error methods
  (t-test, OLS residuals).
- **Publication requirement** — many biomedical journals ask for
  a normality check.

## When NOT to use

- **Formal decision at large n** — tests over-reject with trivial
  departures; the qq-plot is more informative.
- **Non-independence** — normality tests assume iid data.

## Assumptions & caveats

- **KS with estimated parameters** — the standard `scipy.stats.
  kstest` p is anti-conservative; use Lilliefors (`statsmodels.
  stats.diagnostic.lilliefors`).
- **Ties / discrete outcomes** — most tests assume continuous
  data.
- **Report the test used** — different tests can disagree at
  borderline p.
- **Anderson-Darling** is generally most powerful across
  alternatives.

## Related in this repo

- `kolmogorov-smirnov` (implementation of KS), `permutation-tests`
  (nonparametric alternative), `jarque-bera` (if standalone).

## Run

```
python techniques/normality-tests/python/normality_tests.py
Rscript techniques/normality-tests/r/normality_tests.R
```

**Refs:** Shapiro, S.S. & Wilk, M.B. "An analysis of variance test for normality (complete samples)." *Biometrika*, 1965; Stephens, M.A. "EDF statistics for goodness of fit and some comparisons." *JASA*, 1974; Jarque, C.M. & Bera, A.K. "Efficient tests for normality, homoscedasticity and serial independence of regression residuals." *Economics Letters*, 1980.

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

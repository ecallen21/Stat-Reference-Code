# Jarque-Bera Test (Reference §47.103)

Jarque & Bera (1980). Tests normality via SAMPLE SKEWNESS and
KURTOSIS:

    JB = n/6 [ S² + (K − 3)² / 4 ],   JB ~ χ²₂ under H₀.

Fast; the standard normality test on REGRESSION RESIDUALS in
econometrics. Less powerful than Anderson-Darling for pure-tail
departures but easier to compute and interpret.

## Files

- `python/jarque_bera_test.py` — from-scratch JB statistic with
  χ²₂ p-value. Demo (n=500):
  - N(0, 1): JB = 6.58, p = 0.037 (borderline)
  - Uniform: JB = 28.7, p < 10⁻⁴
  - t₃:      JB = 431, p < 10⁻⁴ (heavy tails)
  - Log-normal: JB = 14 945, p < 10⁻⁴ (skew 4.07, kurt 28.5)
  - Bimodal mixture: JB = 57, p < 10⁻⁴.
- `r/jarque_bera_test.R` — `tseries::jarque.bera.test`,
  `moments::jarque.test` (R); `statsmodels`, `scipy.stats.jarque_bera`
  (Python).

## When to use

- **OLS / GLM residual diagnostics** — quick normality check.
- **Financial return distributions** — flag skew / heavy tails.
- **Large n** — asymptotic χ²₂ p-values reliable.
- **Complementary to A-D** — moment-based vs EDF-based views.

## When NOT to use

- **Very small n** — Type-I error inflated; use Monte-Carlo p.
- **When only ONE moment departs** (e.g. skew alone) — JB dilutes
  power; a targeted skew or kurtosis test is stronger.
- **Discrete data** — sample kurtosis unreliable.

## Assumptions & caveats

- **Asymptotic** — χ²₂ only for large n; use small-sample tables
  (Urzúa 1996, Doornik-Hansen).
- **Sensitive to outliers** — a few extreme values inflate JB.
- **Doornik-Hansen** improves p-value calibration under
  contamination.
- **Complements**: KS, A-D, CVM, Shapiro-Wilk — no single "best"
  normality test; report multiple.

## Related in this repo

- `anderson-darling-test`, `cramer-von-mises-test`,
  `kolmogorov-smirnov`, `normality-tests`, `shape-skewness-kurtosis`
  — GoF cousins.
- `qq-plots` (via `regression-diagnostics`) — visual normality
  diagnostic.
- `newey-west-hac`, `sandwich-robust-se` — remedies when residuals
  aren't Gaussian.
- `robust-regression`, `mm-estimators-robust` — robust alternatives
  under fat-tailed errors.

## Run

```
python techniques/jarque-bera-test/python/jarque_bera_test.py
Rscript techniques/jarque-bera-test/r/jarque_bera_test.R
```

**Refs:** Jarque, C.M. & Bera, A.K. "Efficient tests for normality, homoscedasticity and serial independence of regression residuals." *Econ Letters* 6(3): 255-259, 1980; Doornik, J.A. & Hansen, H. "An omnibus test for univariate and multivariate normality." *Oxford Bull Econ Stat* 70(s1): 927-939, 2008.

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

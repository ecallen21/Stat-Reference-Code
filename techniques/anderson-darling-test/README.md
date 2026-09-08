# Anderson-Darling Test (Reference §47.101)

Anderson & Darling (1954). EDF-based goodness-of-fit test that
upweights the TAILS:

    A² = −n − (1/n) Σᵢ (2i−1)(log U₍ᵢ₎ + log(1 − U₍n+1−i₎))

with U₍ᵢ₎ = F₀(x₍ᵢ₎). More powerful than KS for tail departures.

## Files

- `python/anderson_darling_test.py` — from-scratch A² + A²*
  small-sample correction + Marsaglia-Marsaglia asymptotic p-value.
  Demo (n=500):
  - N(0, 1): p = 0.29 (fail to reject)
  - t₃:      p < 10⁻⁴
  - Uniform: p < 10⁻⁴
  - N(0, 1) + 2 % outliers: p < 10⁻⁴.
- `r/anderson_darling_test.R` — `nortest::ad.test`,
  `goftest::ad.test` (R); `scipy.stats.anderson`, from-scratch
  (Python).

## When to use

- **Goodness-of-fit** to a specific distribution.
- **Detecting tail departures** — heavy tails, extreme values,
  outliers.
- **Residual diagnostics** in regression / GLM.
- **Preferred over KS** when the tails matter.

## When NOT to use

- **Comparing two samples** — use Anderson-Darling k-sample or MMD.
- **Very small n** — critical-value tables sparse; use bootstrap p.
- **Fitted params but no correction** — critical values shift when
  parameters are estimated from data.

## Assumptions & caveats

- **IID sample** — dependence inflates apparent evidence.
- **Estimated parameters** require D'Agostino-Stephens corrections
  or Monte-Carlo critical values.
- **Continuous distributions** — discrete data need modifications
  (Choulakian-Stephens).
- **Approximation of p-value** loses accuracy at very small p.

## Related in this repo

- `kolmogorov-smirnov`, `cramer-von-mises-test`,
  `jarque-bera-test`, `normality-tests` — GoF cousins.
- `qq-plots` (via `regression-diagnostics`), `shape-skewness-kurtosis`
  — visual/moment diagnostics.
- `mmd-two-sample-test`, `hsic-independence` — kernel-based
  distributional tests.
- `extreme-value-theory`, `copulas`, `cvar-expected-shortfall` —
  tail-focused analyses.

## Run

```
python techniques/anderson-darling-test/python/anderson_darling_test.py
Rscript techniques/anderson-darling-test/r/anderson_darling_test.R
```

**Refs:** Anderson, T.W. & Darling, D.A. "A test of goodness of fit." *JASA* 49(268): 765-769, 1954; D'Agostino, R.B. & Stephens, M.A. *Goodness-of-Fit Techniques.* Marcel Dekker, 1986.

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

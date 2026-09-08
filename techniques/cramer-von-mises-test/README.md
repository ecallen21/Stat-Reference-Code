# Cramér-von Mises Test (Reference §47.102)

Cramér (1928); von Mises (1931). One-sample EDF-based goodness-of-
fit test:

    W² = n ∫ (F_n(x) − F₀(x))² dF₀(x)
       = 1/(12n) + Σᵢ (U₍ᵢ₎ − (2i−1)/(2n))²

Weights all deviations equally (unlike Anderson-Darling, which
upweights the tails). Anderson (1962) extended to a two-sample W².

## Files

- `python/cramer_von_mises_test.py` — from-scratch W² one-sample
  via `scipy.stats.cramervonmises` for the p-value; separately a
  two-sample T statistic. Demo:
  - N(0, 1) vs Normal:   W² = 0.045,  p ≈ 0.90
  - t₃ vs Normal:         W² = 0.94,  p ≈ 0.003
  - Uniform vs Normal:    W² = 0.72,  p ≈ 0.011
  - Laplace vs Normal:    W² = 1.01,  p ≈ 0.002.
- `r/cramer_von_mises_test.R` — `goftest::cvm.test`, `dgof`,
  `CvM2SL2Test` (R); `scipy.stats.cramervonmises`, from-scratch
  (Python).

## When to use

- **General-purpose GoF** — competitive with A-D when tails matter
  less.
- **Alongside A-D and KS** in a triangulation.
- **Two-sample distribution comparison** — CVM 2-sample variant
  is a common permutation-test baseline.

## When NOT to use

- **When tails dominate** — Anderson-Darling more powerful.
- **Discrete distributions** without adjustment.
- **Very small n** — asymptotic p-value inaccurate; bootstrap.

## Assumptions & caveats

- **IID sample**.
- **Fitted-parameter p-values** shift; use Monte-Carlo cvalues.
- **Choice among KS / CVM / A-D** trades where power sits:
  KS = uniform, CVM = uniform integrated, A-D = tail-weighted.

## Related in this repo

- `anderson-darling-test`, `kolmogorov-smirnov`,
  `jarque-bera-test`, `normality-tests` — GoF cousins.
- `permutation-tests`, `mmd-two-sample-test`, `hsic-independence`
  — distributional-test peers.
- `qq-plots`, `shape-skewness-kurtosis` — visual/moment
  diagnostics.

## Run

```
python techniques/cramer-von-mises-test/python/cramer_von_mises_test.py
Rscript techniques/cramer-von-mises-test/r/cramer_von_mises_test.R
```

**Refs:** Cramér, H. "On the composition of elementary errors." *Skand. Aktuarietidskr.* 11: 13-74, 1928; von Mises, R. *Wahrscheinlichkeitsrechnung.* Deuticke, 1931; Anderson, T.W. "On the distribution of the two-sample Cramér-von Mises criterion." *Ann Math Stat* 33: 1148-1159, 1962.

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

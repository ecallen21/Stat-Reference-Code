# Cauchy Distribution (Reference §47.392)

Heavy-tailed pathological case. PDF:

```
f(x; x₀, γ) = 1 / (π γ (1 + ((x − x₀) / γ)²))
```

- **No mean, no variance** (both integrals diverge).
- Sample mean does NOT converge (violates the Law of Large
  Numbers); use the **MEDIAN**.
- Ratio of two independent `N(0, 1)` random variables is
  Cauchy.
- Foundational counter-example — used to stress-test robust
  estimators and heavy-tail-safe algorithms.

## Files

- `python/cauchy_distribution.py` — Inverse-CDF sampling
  `x = x₀ + γ · tan(π(U − 0.5))`. On n=10 000 draws from
  `Cauchy(2, 1.5)`: sample mean is +4.15 (nonsensical),
  median +2.03 (recovers x₀), IQR-based γ = 1.48 (recovers
  γ), sample variance ~ 1.7e4 (theory: infinite).
  Running-mean of first 10 000 draws ranges over [−0.4,
  +9.7] — a Gaussian would sit within ±0.1 of 2.0.
- `r/cauchy_distribution.R` — `stats::dcauchy` /
  `rcauchy` / `pcauchy` (R); `scipy.stats.cauchy`,
  from-scratch (Python).

## Where else it appears in this repo

- `robust-location-scale`, `huber-m-estimator`,
  `tukey-biweight-m-estimator`, `hampel-identifier` —
  robust estimators designed to survive Cauchy-like tails.
- `hodges-lehmann`, `theil-sen-slope`, `wilcoxon-signed-rank`
  — rank/median statistics unfazed by Cauchy.
- `quantile-regression`, `censored-quantile-regression` —
  median-quantile regression as heavy-tail defence.
- `extreme-value-theory` — Cauchy is a Fréchet-domain
  (max-stable) distribution.
- `bayesian-linear-regression` — Cauchy priors on
  coefficients as a Student-t special case (Gelman's default).

## Assumptions & caveats

- **Estimator choice** — plug-in mean is invalid; ALWAYS use
  median / MLE.
- **MLE** — no closed form; Newton on the log-likelihood.
  Fisher information matrix well defined and finite.
- **Confidence intervals** — bootstrap works (via medians),
  but percentile bootstrap on the mean does not.
- **Multivariate Cauchy** — Student-t with 1 df; heavy-
  tailed and closed under linear transforms.
- **Simulation stress test** — a common sanity check is
  running your pipeline on Cauchy data and asserting the
  estimator does not blow up.

## Run

```
python techniques/cauchy-distribution/python/cauchy_distribution.py
Rscript techniques/cauchy-distribution/r/cauchy_distribution.R
```

**Refs:** Feller, W. *An Introduction to Probability Theory and Its Applications*, Vol II, 2nd ed., Wiley, 1971.

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

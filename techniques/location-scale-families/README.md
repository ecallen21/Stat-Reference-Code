# Location-Scale Families (Reference §47.398)

`X = μ + σ · Z` where `Z` follows a fixed standard
distribution. All family members share:

```
F(x; μ, σ) = F₀((x − μ)/σ)
f(x; μ, σ) = (1/σ) · f₀((x − μ)/σ)
```

Common members: Normal, Cauchy, Student-t, Laplace, logistic,
Gumbel, uniform on `[0, 1]`. Fisher information matrix is
DIAGONAL in the invariant `(μ, log σ)` parameterisation for
symmetric families.

## Files

- `python/location_scale_families.py` — three symmetric
  members at `(μ=2, σ=1.5)` on n=10 000: Normal gives
  reliable mean/sd; Laplace shows sd is biased upward
  (Laplace variance is `2σ²`) while MAD and IQR-scaled both
  recover σ = 1.5; Cauchy's mean explodes and sd is
  meaningless, but median = 1.97 recovers μ (IQR-scaled
  reports 2.22 because Cauchy IQR is `2γ`, not `1.349 γ` —
  the Gaussian-calibration factor doesn't fit).
- `r/location_scale_families.R` — `scipy.stats` `loc`/`scale`,
  `stats::dnorm/rnorm/...`, base R.

## Where else it appears in this repo

- `robust-location-scale`, `huber-m-estimator`,
  `tukey-biweight-m-estimator`, `hampel-identifier` —
  robust estimators for location-scale families.
- `cauchy-distribution`, `weibull-distribution`,
  `lognormal-distribution` — specific members.
- `quantile-regression`, `censored-quantile-regression`,
  `bayesian-quantile-regression` — quantile-based inference
  respects location-scale invariance.
- `mm-estimators-robust`, `regression-diagnostics` —
  scale-equivariance is a design principle.

## Assumptions & caveats

- **Invariant estimators** — an estimator `T(X)` is
  location-scale equivariant if `T(aX + b) = a T(X) + b`;
  median and IQR are the archetypes.
- **MLE** — closed form for Normal (mean, sd); iterative /
  M-estimator for Cauchy, Laplace, Student-t.
- **Scale-calibration** — MAD × 1.4826 is Fisher-consistent
  at Normal only; other members have their own
  calibration constants.
- **Log-location-scale** — `Y = log X` on positive support
  gives log-Normal / Weibull / log-logistic; useful for
  survival modelling.
- **Fisher information matrix diagonal** — enables
  independent-parameter inference under symmetric
  standardisation.

## Run

```
python techniques/location-scale-families/python/location_scale_families.py
Rscript techniques/location-scale-families/r/location_scale_families.R
```

**Refs:** Casella, G. and Berger, R.L. *Statistical Inference*, 2nd ed., Duxbury, 2002, §3.5.

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

# Nonparametric Bootstrap — Case Resampling (Reference §10.1)

Given a sample `x₁..x_n` and a statistic `θ̂ = T(x₁..x_n)`, estimate the sampling distribution of `θ̂` by resampling the **data** with replacement `B` times, computing `T` on each resample, and using the empirical distribution of the resulting `θ*_1..θ*_B`.

This is the workhorse of nonparametric inference — no distributional assumption on `x`, works for any statistic that's a function of the sample.

## Confidence intervals covered here

| CI type | Formula | Notes |
|---|---|---|
| **Percentile** | `[Q_{α/2}(θ*), Q_{1−α/2}(θ*)]` | Simplest; transformation-respecting |
| **Basic (pivotal)** | `[2θ̂ − Q_{1−α/2}, 2θ̂ − Q_{α/2}]` | Better small-sample behavior when `θ̂` is biased |
| **Normal** | `θ̂ ± z · SE_bootstrap` | Assumes approximately-normal sampling dist |

For **BCa** (bias- and skewness-adjusted; usually the best default) and a side-by-side comparison of methods, see [`bca-bootstrap`](../bca-bootstrap) (§10.3 + §10.14).

## Files

- `python/nonparametric_bootstrap.py` — 1-D and 2-D (row-resampling) bootstrap; percentile / basic / normal CIs; SE. Percentile CI and SE match `scipy.stats.bootstrap` closely on the demo.
- `r/nonparametric_bootstrap.R` — from-scratch + `boot::boot` / `boot::boot.ci`.
- `pyspark/nonparametric_bootstrap.py` — Spark DataFrame case resampling via `df.sample(withReplacement=True, fraction=1.0)`. Useful when the raw data is too big to collect.

## Assumptions

- Observations are IID (case resampling assumption). For dependent data (time series) use [`block-bootstrap`](../block-bootstrap).
- For regression with heteroscedasticity, [`wild-bootstrap`](../wild-bootstrap) is often preferable.
- Enough distinct values to make resampling meaningful — with `n = 5` and any statistic, the bootstrap distribution is discrete and jagged.

## Run

```
python techniques/nonparametric-bootstrap/python/nonparametric_bootstrap.py
Rscript techniques/nonparametric-bootstrap/r/nonparametric_bootstrap.R
python techniques/nonparametric-bootstrap/pyspark/nonparametric_bootstrap.py
```

**Refs:** Efron, B. "Bootstrap methods: Another look at the jackknife." *Ann. Stat.* 7(1), 1–26, 1979; Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap*, Chapman & Hall, 1993; Davison, A.C. & Hinkley, D.V. *Bootstrap Methods and Their Application*, Cambridge, 1997.

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

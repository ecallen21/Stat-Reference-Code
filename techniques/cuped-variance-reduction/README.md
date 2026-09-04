# CUPED — Controlled-experiment Using Pre-Experiment Data (Reference §44.3)

Deng, Xu, Kohavi & Walker (2013). Regression adjustment using a
**pre-experiment covariate** X to reduce the variance of the
outcome Y in an A/B test.

```
Y_cuped = Y − θ · (X − X̄)     with θ = Cov(Y, X) / Var(X)
```

Because assignment is randomised independently of X, the estimator
is unbiased and the variance drops by roughly `1 − ρ²` where
`ρ = corr(Y, X)`. Effective sample size grows by `1 / (1 − ρ²)`.

## When to use

- **Repeat-user products** where each user has a comparable
  pre-experiment metric (last week's spend, prior sessions).
- **High-variance metrics** where a modest correlation with a
  pre-period covariate can double effective sample size.

## When NOT to use

- **First-time users** at time zero (no pre-period).
- **Metrics uncorrelated with any available pre-period covariate**.
- **Assignment correlated with covariate** — CUPED is unbiased
  only under random assignment.

## Files

- `python/cuped_variance_reduction.py` — pooled-θ CUPED adjustment
  + Welch t before / after. Demo (n=5000/arm, ρ≈0.8): **variance
  reduction 64 %, effective n multiplier 2.75×**; SE drops from
  0.020 → 0.012 while the treatment effect is preserved.
- `r/cuped_variance_reduction.R` — `stats::lm` regression
  adjustment, `sandwich` (R); `statsmodels.OLS`, custom (Python).

## Assumptions & caveats

- **Pooled θ** across arms (Deng 2013) — not per-arm; avoids
  regression-to-the-mean bias.
- **Covariate observed pre-experiment** — a covariate measured
  during the experiment risks biasing the estimate.
- **Non-linear alternatives** (CUPAC / CUPED with GBM) — capture
  more variance at the cost of complexity.
- **Report both raw and CUPED-adjusted** effect sizes and SEs.

## Related in this repo

- `ab-test-fundamentals` — the underlying tests CUPED plugs into.
- `mde-sample-size` — CUPED lets you halve required n.
- `regression-adjustment` (implicit) — CUPED is a special case.

## Run

```
python techniques/cuped-variance-reduction/python/cuped_variance_reduction.py
Rscript techniques/cuped-variance-reduction/r/cuped_variance_reduction.R
```

**Refs:** Deng, A., Xu, Y., Kohavi, R., & Walker, T. "Improving the sensitivity of online controlled experiments by utilizing pre-experiment data." *WSDM*, 2013; Poyarkov, A., Drutsa, A., Khalyavin, A., Gusev, G., & Serdyukov, P. "Bootstrapped and stratified approaches in A/B testing." *SIGIR*, 2016.

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

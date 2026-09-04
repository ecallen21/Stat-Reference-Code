# A/B Test Fundamentals (Reference §44.1)

Kohavi-Tang-Xu (2020). Randomised controlled experiment comparing
two (or more) variants online. Report **lift + confidence interval
+ p-value** for the target metric using the appropriate two-sample
test.

## Standard tests

- **Binary outcome** (conversion, click-through) — two-proportion
  z-test.
- **Continuous outcome** (revenue, time-on-page) — Welch's
  two-sample t-test.
- **Categorical / count outcome** — chi-square or exact test.

## When to use

- **Every product experiment** with a randomised control /
  treatment split.
- **Baseline analysis** before considering variance-reduction
  (CUPED) or sequential monitoring.

## When NOT to use

- **Non-independent units** (network / clustered) — see
  `interference-cluster`.
- **Continuous monitoring** without correction — see
  `always-valid-inference`.
- **Ratio metrics** with correlated numerator/denominator — use
  the delta method (see `ratio-metrics-abtest`).

## Files

- `python/ab_test_fundamentals.py` — two-proportion z + Welch t
  from scratch. Demo: conversion 5.0 % → 5.5 % on n=10 000/arm
  gives **abs lift 0.005 (95% CI [−0.001, 0.011], p=0.11)** — not
  significant; revenue-per-user Gamma(2, 5) → Gamma(2, 5.4) gives
  **lift 0.72 (95% CI [0.43, 1.01], p=1.1×10⁻⁶)**.
- `r/ab_test_fundamentals.R` — `stats::t.test`, `prop.test`,
  `chisq.test`, `pwr` (R); `scipy.stats.ttest_ind`,
  `chi2_contingency`, `statsmodels.stats.proportion` (Python).

## Assumptions & caveats

- **Random assignment** — confirm via covariate balance and Sample
  Ratio Mismatch (SRM) checks before interpreting results.
- **Fixed horizon** — do not stop early on p; use sequential
  methods if peeking.
- **Metric selection** pre-registered — post-hoc metric hunting
  inflates false positives.
- **Unit of randomisation matches unit of analysis** — a user-
  randomised experiment reporting per-session results is inflated.

## Related in this repo

- `mde-sample-size` — plan the experiment.
- `cuped-variance-reduction` — tighten CIs.
- `always-valid-inference` — peek-safe monitoring.
- `multiple-metrics-fdr` — correct across metrics.

## Run

```
python techniques/ab-test-fundamentals/python/ab_test_fundamentals.py
Rscript techniques/ab-test-fundamentals/r/ab_test_fundamentals.R
```

**Refs:** Kohavi, R., Tang, D., & Xu, Y. *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*, Cambridge University Press, 2020; Kohavi, R., Longbotham, R., Sommerfield, D., & Henne, R.M. "Controlled experiments on the web: survey and practical guide." *Data Mining and Knowledge Discovery*, 2009.

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

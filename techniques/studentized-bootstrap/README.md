# Studentized (Bootstrap-t) Confidence Intervals (Reference §10.4)

Refinement of the percentile bootstrap that uses a **pivotal** quantity:

```
t^*_b = (θ̂^*_b − θ̂) / se(θ̂^*_b)
```

Take the α/2 and 1 − α/2 quantiles of `t^*` and back out the CI:

```
CI = (θ̂ − t^*_{1−α/2} · se(θ̂),   θ̂ − t^*_{α/2} · se(θ̂))
```

Because `t^*` is asymptotically pivotal (its distribution does not depend on `θ`), the bootstrap-t is **second-order accurate** (Hall 1988): coverage error is `O(1/n)`, versus `O(1/√n)` for the plain percentile bootstrap. In skewed samples this often matters.

## Estimating `se(θ̂^*)`

- **Analytical**: closed-form SE (e.g. `s / √n` for the mean, delta-method SE for MLEs).
- **Nested bootstrap**: inner bootstrap of each resample to estimate its SE. Expensive (`B · B_inner` total resamples) but always available.

## Files

- `python/studentized_bootstrap.py` — bootstrap-t for the mean using analytical SE, plus the nested-bootstrap variant for arbitrary statistics (demonstrated on the median). Demo on n = 40 lognormal sample: bootstrap-t 95% CI for the mean = (0.98, 1.67); nested-bootstrap-t CI for the median = (0.53, 1.05).
- `r/studentized_bootstrap.R` — `boot::boot(...)` where the statistic returns both `θ̂` and its variance, then `boot::boot.ci(type = "stud")`.

## When to use

- Skewed statistics (mean of a heavy-tailed sample, ratio of means, correlation) where the sampling distribution is asymmetric — bootstrap-t adapts to the asymmetry, percentile bootstrap does not.
- When a closed-form SE is cheaply available; nested-bootstrap-t is a heavy hammer, mostly reserved for statistics where BCa (bias-corrected accelerated) is impractical.

## Contrast with other bootstrap CIs

| Method                | Order accuracy | Needs SE? | Best for                              |
|-----------------------|----------------|-----------|---------------------------------------|
| Percentile            | first-order    | no        | quick, roughly symmetric distributions |
| Basic                 | first-order    | no        | same, slightly less biased            |
| **Bootstrap-t**       | **second-order** | yes     | skewed sampling distributions          |
| BCa                   | second-order   | jackknife-based `a` | non-monotone transforms       |

## Assumptions & caveats

- **SE must be reliably estimable** — an unstable inner SE gives unstable `t^*` and a lopsided CI.
- **B large enough**: `B ≥ 999` for CIs; nested bootstrap `B_inner ≥ 50`.
- **Extreme tail behavior**: the ratio `(θ̂^* − θ̂) / se(θ̂^*)` can have very heavy tails when `se(θ̂^*)` occasionally collapses; winsorize or use BCa when this happens.

## Run

```
python techniques/studentized-bootstrap/python/studentized_bootstrap.py
Rscript techniques/studentized-bootstrap/r/studentized_bootstrap.R
```

**Refs:** Efron, B. & Tibshirani, R.J. *An Introduction to the Bootstrap*, Chapman & Hall, 1993 (Ch 12); Hall, P. "Theoretical comparison of bootstrap confidence intervals." *Ann. Stat.* 16(3), 927–953, 1988; Davison, A.C. & Hinkley, D.V. *Bootstrap Methods and Their Application*, Cambridge, 1997.

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

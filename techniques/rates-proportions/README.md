# Summary Statistics for Rates & Proportions (Reference §1.8)

How often an event occurs in a population — and how precisely we know it.

## Point estimates
| Measure | Definition | Design |
|---------|-----------|--------|
| Prevalence | proportion **with** a condition at a point in time | cross-sectional |
| Incidence proportion ("risk") | proportion **developing** the condition over a period | cohort, fixed follow-up |
| Incidence rate | events ÷ **person-time at risk** | cohort, variable follow-up |
| Person-time | Σ of individual at-risk follow-up durations | denominator for the rate |

`p/(1−p)` is the **odds**; the instantaneous version of an incidence rate is the **hazard**. Standardized rates (age-adjustment etc.) come later (Ch. 21).

## Confidence intervals for a proportion `p = x/n`
| Method | Formula / basis | Notes |
|--------|-----------------|-------|
| Wald (normal approx) | `p ± z·√(p(1−p)/n)` | Poor near 0/1 and for small `n` — avoid |
| Wilson score | inverts the score test | **Recommended default**; well-behaved at the extremes |
| Clopper–Pearson (exact) | Beta-distribution quantiles | Guaranteed ≥ nominal coverage (conservative) |
| Agresti–Coull | Wald on `(x+z²/2)/(n+z²)` | Simple, nearly as good as Wilson |

## Confidence interval for a rate `λ = events / person_time`
**Exact Poisson** (Garwood): lower `= χ²_{α/2, 2e}/2 / PT`, upper `= χ²_{1−α/2, 2e+2}/2 / PT`. Reduces to a sensible one-sided bound when `events = 0`.

## Files
- `python/rates_proportions.py` — from-scratch point estimates + Wald / Wilson / Clopper–Pearson / exact-Poisson CIs; compares against `statsmodels.stats.proportion.proportion_confint`
- `r/rates_proportions.R` — from-scratch + base `binom.test` (Clopper–Pearson), `prop.test` (Wilson), `poisson.test` (exact rate CI); notes on `epitools`, `binom`
- `pyspark/rates_proportions.py` — distributed `groupBy` aggregation of cases / person-time by stratum, then the closed-form Wilson / Poisson CIs on the driver

## Run
```
python techniques/rates-proportions/python/rates_proportions.py
Rscript techniques/rates-proportions/r/rates_proportions.R
python techniques/rates-proportions/pyspark/rates_proportions.py
```

**Ref:** Rothman, Greenland & Lash, *Modern Epidemiology*, 4th ed., Wolters Kluwer, 2021.

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

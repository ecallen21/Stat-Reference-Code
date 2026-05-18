# Marginal Effects for GLMs (Reference §7.37, §7.47)

A GLM coefficient `β_j` lives on the **link scale** (log-odds for logistic, log-rate for Poisson, …). Often the substantive question is "what's the effect on `y` — on the *response* scale — for a unit increase in `x_j`?"

That's a derivative (or discrete change) `∂μ/∂x_j` rather than `∂η/∂x_j = β_j`.

## Three flavors

| Flavor | Definition | When to use |
|--------|-----------|-------------|
| **AME** (Average Marginal Effect) | Mean over observations of `∂μ_i/∂x_j` evaluated at each `x_i` | **Default.** Reports an "average effect across the sample." |
| **MEM** (Marginal Effect at the Means) | `∂μ/∂x_j` evaluated at `x = mean(x)` | Easier by hand; misleading when the mean point isn't representative (e.g. mean of a binary variable is fractional and meaningless) |
| **MER** (Marginal Effect at Representative values) | `∂μ/∂x_j` evaluated at a chosen scenario (e.g. men, age 60, region X) | "What's the effect for a person who looks like this?" |

For **binary predictors** the marginal effect is a **discrete difference**:
`μ(x_j = 1) − μ(x_j = 0)`, with other covariates held at chosen values (often averaged across the sample, giving "average discrete change").

## Derivatives of the link

| Family / link | `∂μ/∂x_j` |
|---------------|------------|
| Logit (logistic) | `π·(1 − π)·β_j` |
| Probit | `φ(η)·β_j` |
| Log link (Poisson, Gamma) | `μ·β_j` |

For a 0.30 baseline probability in logistic, AME with `β = 0.8` is about `0.21 × 0.8 = 0.168` — far from 0.8 itself.

## Why this matters in reporting

- **Odds ratios** are unitless and stable but hard for clinical / policy audiences to interpret. AMEs are in the same units as the outcome (probability points for binary, expected count for Poisson) and translate directly to "10 percentage points more X causes a Y-point change in outcome."
- Most regression reporting guidelines now recommend reporting **both** the coefficient (for the methodologically inclined) and the marginal effects (for the substantive interpretation).

## Files
- `python/marginal_effects.py` — AME / MEM for logit, probit, Poisson; discrete AME for binary predictors; cross-check pattern via `statsmodels.Logit.get_margeff`.
- `r/marginal_effects.R` — from-scratch + notes on `margins::margins` and `marginaleffects::avg_slopes` (the canonical modern implementations).
- PySpark: N/A — post-fit transform on small parameters.

## Run
```
python techniques/marginal-effects/python/marginal_effects.py
Rscript techniques/marginal-effects/r/marginal_effects.R
```

**Refs:** Greene, *Econometric Analysis*, 8th ed., 2018; Norton & Dowd, "Log Odds and the Interpretation of Logit Models," *Health Services Research* 53(2), 859–878, 2018.

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

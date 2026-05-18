# Modified Poisson Regression for Risk Ratios (Reference §7.9, §7.53)

A trick for getting **risk ratios** from **binary** outcomes when the outcome is common (prevalence > ~10%) and you'd rather not use log-binomial (which often fails to converge).

## The trick (Zou 2004)

1. Fit Poisson regression with **log link** on the binary outcome:

   `log E[yᵢ] = x'ᵢβ`,   even though `yᵢ ∈ {0, 1}`.

2. The point estimates `β̂` are **valid log-RR estimates** — Poisson is consistent for the mean structure even when the variance assumption is wrong.

3. Replace the Poisson SEs (which assume `Var = mean`) with **Huber–White sandwich SEs**:

   `Var_robust(β̂) = (X'WX)⁻¹ (X' · diag(eᵢ²) · X) (X'WX)⁻¹`

   with `W = diag(μᵢ)` and `eᵢ = yᵢ − μᵢ`. (The HC1 small-sample correction multiplies by `n/(n−p)`.)

## Why this matters in epidemiology

The **odds ratio overstates the risk ratio** when the outcome is common:
- Rare outcome (~1% prevalence): OR ≈ RR. Use logistic.
- Common outcome (>10% prevalence): OR > RR (sometimes 2–3×). Logistic ORs become hard to communicate to clinical audiences.

Two clean ways to get RRs:
- **Log-binomial** (`Y ~ Bernoulli, log link`): the "right" model, but the linear predictor `x'β` can give fitted probabilities > 1, causing convergence problems.
- **Modified Poisson** (this file): no convergence issues, valid estimates, robust SEs.

`epitools`, `epicalc`, and `survey::svyglm` in R all use this pattern. The **CONSORT** guidelines (and many journals) prefer RRs to ORs for trial reporting when outcomes are common.

## Files
- `python/modified_poisson.py` — from-scratch Poisson IRLS + Huber–White sandwich; matches `statsmodels.GLM(family=Poisson(), cov_type='HC1')` to 3 decimals.
- `r/modified_poisson.R` — from-scratch + `sandwich::vcovHC(type = "HC1")` cross-check via `lmtest::coeftest`.
- PySpark: N/A — sandwich SEs at Spark scale require holding the residuals in memory; sample and use the Python version for moderate `n`.

## Run
```
python techniques/modified-poisson/python/modified_poisson.py
Rscript techniques/modified-poisson/r/modified_poisson.R
```

**Refs:** Zou, "A Modified Poisson Regression Approach to Prospective Studies with Binary Data," *American Journal of Epidemiology* 159(7), 702–706, 2004; Wacholder, "Binomial Regression in GLIM: Estimating Risk Ratios and Risk Differences," *American Journal of Epidemiology* 123(1), 174–184, 1986.

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

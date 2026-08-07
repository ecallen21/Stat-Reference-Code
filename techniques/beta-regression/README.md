# Beta Regression (Reference §5.20)

Continuous outcomes strictly in `(0, 1)` — proportions, rates, percentages divided by 100 — have variance that shrinks near 0 and 1. OLS on the raw scale (or on the logit) is a workaround; Beta regression handles it cleanly via a **Beta distribution**.

## Ferrari–Cribari-Neto (2004) parameterization

```
y ~ Beta(μ · φ, (1 − μ) · φ)
mean:  μ ∈ (0, 1)
precision:  φ > 0    with  Var(y) = μ (1 − μ) / (1 + φ)
```

Two submodels:

- **Mean**: `logit(μ_i) = X_i β`
- **Precision** (optional): `log(φ_i) = Z_i γ` — variable dispersion. If omitted, `φ` is a constant.

## Boundary values

`y = 0` or `y = 1` don't fit inside the Beta support. Options:

- **Smithson-Verkuilen transform**: `y' = (y (n − 1) + 0.5) / n`.
- **Zero / one / zero-and-one inflated Beta** — mixture with point masses at the boundaries.
- **Drop boundary cases** — only if there are few.

## Files

- `python/beta_regression.py` — from-scratch BFGS MLE with constant or variable-precision submodel. Demo (n = 400, true β = 1.2, φ = 20): recovers β = 1.196, φ = 19.8; variable-precision variant recovers γ = (3.00, 0.50) matching truth (3.0, 0.5).
- `r/beta_regression.R` — `betareg::betareg` (standard R package, supports precision submodels, link options, and hypothesis tests).

## When to use

- Proportion or rate outcomes in `(0, 1)` — vote shares, forest cover, disease prevalence per site.
- Sub-100 percentages — patient adherence, response quality scores.
- Whenever residuals from a linear model on a bounded outcome are systematically funnel-shaped near the boundaries.

## Assumptions & caveats

- **All observations strictly in `(0, 1)`** — handle boundaries as noted.
- **Independence** — extend to random effects with `glmmTMB(y ~ x + (1 | id), family = beta_family())`.
- **Link functions**: logit (default), probit, cloglog, loglog available in `betareg`.
- **Precision heterogeneity**: fit the variable-precision model even if you don't care about `γ` — a bad constant-`φ` fit can look like mean misspecification.

## Run

```
python techniques/beta-regression/python/beta_regression.py
Rscript techniques/beta-regression/r/beta_regression.R
```

**Refs:** Ferrari, S.L.P. & Cribari-Neto, F. "Beta regression for modelling rates and proportions." *J. Appl. Stat.* 31(7), 799–815, 2004; Smithson, M. & Verkuilen, J. "A better lemon squeezer? Maximum-likelihood regression with beta-distributed dependent variables." *Psychol. Methods* 11(1), 54–71, 2006.

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

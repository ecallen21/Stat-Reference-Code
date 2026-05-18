# Poisson Regression (Reference §7.12, §7.43)

Model **count outcomes** with a log link:

`Yᵢ ~ Poisson(μᵢ)`,   `log(μᵢ) = x'ᵢβ + offsetᵢ`

`exp(βⱼ)` is the **incidence rate ratio (IRR)** for `xⱼ`.

## Offset for rate modeling (§7.43)

When the natural quantity is a **rate** (events per unit exposure / person-time / area), include `offsetᵢ = log(exposureᵢ)` so the fitted `μᵢ = exposureᵢ · exp(x'ᵢβ)` represents that rate scaling.

Common offsets:
- `log(person_years)` for incidence rates.
- `log(area)` for spatial counts.
- `log(impressions)` for click-through models.

R: `glm(y ~ x + offset(log(exposure)), family = poisson)`.
Python: `sm.GLM(y, X, family=Poisson(), offset=np.log(exposure))`.
Spark: `GeneralizedLinearRegression(..., offsetCol="log_exposure")`.

## Fitting via IRLS

Standard GLM IRLS with `wᵢ = μᵢ` (Poisson variance = mean) and working response `zᵢ = ηᵢ + (yᵢ − μᵢ)/μᵢ`.

## The overdispersion problem

Poisson assumes `Var(Y) = E[Y]`. Real data often have `Var(Y) > E[Y]` — **overdispersion** — caused by unobserved heterogeneity, clustering, or excess zeros. Symptoms:
- **Pearson dispersion** `χ²_P / (n − p) ≫ 1` (the output flags this).
- Coefficient SEs that look implausibly small; many "significant" findings.

What to do:
- **Negative binomial regression** (`techniques/negative-binomial-regression`) — adds a dispersion parameter.
- **Quasi-Poisson** — same IRR, but scale up the SEs by `√dispersion`. (Available as `glm(family = quasipoisson)`.)
- **Modified Poisson** (`techniques/modified-poisson`) — Poisson with robust (sandwich) SEs; the right tool for **risk ratios from binary outcomes**.
- **Zero-inflated / hurdle** if the data have a big spike at zero (deferred).

A short decision guide (§7.58): start with Poisson → check dispersion → NB if `dispersion > ~1.5` → ZI/Hurdle if zeros are excessive.

## Files
- `python/poisson_regression.py` — from-scratch IRLS with offset support, deviance, Pearson dispersion, AIC, IRR + 95% CI; matches `statsmodels.GLM(family=Poisson())` exactly.
- `r/poisson_regression.R` — from-scratch + base `glm(family = poisson)` with the `offset(...)` term inside the formula.
- `pyspark/poisson_regression.py` — `pyspark.ml.regression.GeneralizedLinearRegression(family="poisson", link="log", offsetCol=...)`.

## Run
```
python techniques/poisson-regression/python/poisson_regression.py
Rscript techniques/poisson-regression/r/poisson_regression.R
python techniques/poisson-regression/pyspark/poisson_regression.py
```

**Refs:** Cameron & Trivedi, *Regression Analysis of Count Data*, 2nd ed., Cambridge UP, 2013; McCullagh & Nelder, *Generalized Linear Models*, 2nd ed., 1989.

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

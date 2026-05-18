# Gamma Regression (Reference §7.25)

GLM for continuous **strictly positive** outcomes (costs, lengths of stay, durations, biomarker concentrations):

`Yᵢ ~ Gamma(shape = 1/φ, scale = φ·μᵢ)`, with `E[Y] = μ`, `Var[Y] = φ·μ²`.

The log link is the standard choice in healthcare / epi:

`log μᵢ = x'ᵢβ`,   `exp(βⱼ)` is the multiplicative effect on the mean.

## Why Gamma over alternatives

| Approach | What it models | Drawback |
|----------|----------------|----------|
| **Gamma GLM (log link)** | `E[Y]` directly on the original scale | Right-tail variance grows with mean²; check fit |
| OLS on `log(Y)` | median (geometric mean) — `E[exp(log Y)] ≠ exp(E[log Y])` | Hard to back-transform to the mean cleanly |
| Inverse-Gaussian | Even heavier right tail than Gamma | Less common; convergence trickier |
| Tweedie (§7.26) | Generalizes Gamma + Poisson (semi-continuous) | Adds a power parameter to estimate |

For **cost data** where you want predicted mean cost, Gamma + log is the standard textbook choice. For **medians** instead of means, quantile regression (a different family) is the better tool.

## Fitting

Standard GLM IRLS. For the log link the weight collapses to 1 (the mean cancels in the weight expression), so each iteration is just OLS on the working response `zᵢ = ηᵢ + (yᵢ − μᵢ)/μᵢ`. The dispersion `φ` is estimated from Pearson `χ²/(n − p)` and scales the coefficient SEs.

## Files
- `python/gamma_regression.py` — from-scratch IRLS + dispersion estimate + deviance; matches `statsmodels.GLM(family=Gamma(link=Log()))` to 12 decimals.
- `r/gamma_regression.R` — from-scratch + base `glm(family = Gamma(link = "log"))`.
- `pyspark/gamma_regression.py` — `pyspark.ml.regression.GeneralizedLinearRegression(family="gamma", link="log")`.

## Run
```
python techniques/gamma-regression/python/gamma_regression.py
Rscript techniques/gamma-regression/r/gamma_regression.R
python techniques/gamma-regression/pyspark/gamma_regression.py
```

**Refs:** McCullagh & Nelder, *Generalized Linear Models*, 2nd ed., 1989; Manning, "The Logged Dependent Variable, Heteroscedasticity, and the Retransformation Problem," *J. Health Economics* 17(3), 283–295, 1998.

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

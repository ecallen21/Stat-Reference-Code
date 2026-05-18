# Probit Regression (Reference §7.10)

Binary GLM with the **probit** link:

`P(y = 1 | x) = Φ(x'β)`,   where Φ is the standard normal CDF.

## Latent-variable view

`y = 1 iff y* = x'β + ε > 0`, with `ε ~ N(0, 1)`. (Logistic uses `ε ~ Logistic(0, 1)` instead.) The latent threshold model is why probit is the default in econometrics — researchers often have a substantive story about the latent `y*` being normally distributed.

## Fitting via IRLS

Same structure as logistic but the weights/working-response use the normal PDF:
- `wᵢ = φ(ηᵢ)² / (Φ(ηᵢ)(1 − Φ(ηᵢ)))`
- `zᵢ = ηᵢ + (yᵢ − Φ(ηᵢ))/φ(ηᵢ)`
- Update: `β ← (X'WX)⁻¹X'Wz`

Matches `statsmodels.Probit` / R's `glm(family = binomial(link = "probit"))` exactly.

## Probit vs. logit

- **Coefficient magnitudes**: probit βs are roughly **logit βs / 1.6** (because the logistic distribution has heavier tails than the standard normal).
- **No clean OR interpretation.** Probit coefficients are in standard-deviation units of the latent `y*`. Report **average marginal effects** instead — `AME_j = φ̄ · β_j` (mean of normal PDF at fitted ηᵢ, times β_j). See `techniques/marginal-effects` for the longer story.
- **Predictions are nearly identical** for any practical purpose. The choice is mostly a tradition / interpretation preference.

## Files
- `python/probit_regression.py` — from-scratch IRLS + AME; matches `statsmodels.Probit`.
- `r/probit_regression.R` — from-scratch + base `glm(family = binomial(link = "probit"))`.
- PySpark: N/A — Spark MLlib's `GeneralizedLinearRegression` doesn't expose probit as a stock link. Logistic is the practical choice at scale.

## Run
```
python techniques/probit-regression/python/probit_regression.py
Rscript techniques/probit-regression/r/probit_regression.R
```

**Refs:** Bliss, "The Calculation of the Dosage-Mortality Curve," *Annals of Applied Biology* 22, 134–167, 1935; McCullagh & Nelder, *Generalized Linear Models*, 2nd ed., 1989.

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

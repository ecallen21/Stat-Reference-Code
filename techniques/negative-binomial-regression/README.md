# Negative Binomial Regression (Reference §7.13)

For **overdispersed** count outcomes (`Var(Y) > E[Y]`), where Poisson is wrong:

`Yᵢ ~ NB(μᵢ, θ)`,   `log(μᵢ) = x'ᵢβ + offsetᵢ`
`Var(Yᵢ) = μᵢ + μᵢ² / θ`

`θ > 0` is the **dispersion parameter**; as `θ → ∞`, NB collapses to Poisson. The reciprocal `α = 1/θ` is also commonly reported (statsmodels uses `α`; R's `MASS::glm.nb` reports `θ`).

## When to use NB instead of Poisson

Symptoms of overdispersion in a Poisson fit:
- Pearson dispersion `χ²_P / (n − p) ≫ 1` (e.g. > 1.5).
- Coefficient SEs look implausibly small; many "significant" findings.
- The score test for overdispersion rejects (see `techniques/overdispersion-tests`).

NB **doesn't change the point estimates** much vs. Poisson — but it scales the SEs up to reflect the true variability, and it gives a proper likelihood (so AIC, LRTs, etc. are valid).

## Fitting

Joint MLE on `(β, log θ)` via BFGS — more numerically robust than alternating IRLS-for-β / 1-D-MLE-for-θ, which is sensitive to the initial `θ`. The R workhorse `MASS::glm.nb` uses an iterative IRLS + Brent's-method scheme; both give the same answer.

## Files
- `python/negative_binomial_regression.py` — joint BFGS MLE on (β, log θ); matches `statsmodels.NegativeBinomial` to ≥ 4 significant figures. Demo recovers `θ ≈ 2.24` from true `θ = 2.0`.
- `r/negative_binomial_regression.R` — joint BFGS + `MASS::glm.nb` cross-check.
- PySpark: N/A — Spark's `GeneralizedLinearRegression` does not ship NB as a stock family. For massive overdispersed counts, use quasi-Poisson (Poisson IRRs + sandwich SEs) or sample and use the Python version.

## Run
```
python techniques/negative-binomial-regression/python/negative_binomial_regression.py
Rscript techniques/negative-binomial-regression/r/negative_binomial_regression.R
```

**Refs:** Cameron & Trivedi, *Regression Analysis of Count Data*, 2nd ed., 2013; Hilbe, *Negative Binomial Regression*, 2nd ed., Cambridge UP, 2011.

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

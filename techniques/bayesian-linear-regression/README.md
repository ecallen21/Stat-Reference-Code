# Bayesian Linear Regression (Reference §14.10, §14.11)

Linear model `y | X, β, σ² ~ Normal(Xβ, σ² I)` with a Normal-Inverse-Gamma conjugate prior.

## Prior

```
β | σ²    ~ Normal(m₀, σ² V₀)
σ²        ~ InvGamma(a₀, b₀)
```

## Joint Normal-Inverse-Gamma posterior (Zellner 1971)

```
V_n⁻¹ = V₀⁻¹ + XᵀX
m_n   = V_n (V₀⁻¹ m₀ + Xᵀ y)
a_n   = a₀ + n/2
b_n   = b₀ + ½ (yᵀy + m₀ᵀ V₀⁻¹ m₀ − m_nᵀ V_n⁻¹ m_n)
```

Marginal posterior of `β` is a multivariate `t` with `2a_n` df and scale `(b_n / a_n) V_n`. Posterior predictive at a new `x*` is univariate `t`.

## Two default choices for `V₀`

- **Zellner g-prior**: `V₀ = g (XᵀX)⁻¹`. Shrinks the OLS estimator by `g / (g + 1)`; letting `g → ∞` recovers OLS. Common defaults: `g = n` (unit-information prior) or `g = p²`.
- **Ridge-like**: `V₀ = τ² I`. MAP estimator equals ridge regression at penalty `λ = σ² / τ²`; the Bayesian version adds full posterior uncertainty.

## Files

- `python/bayesian_linear_regression.py` — closed-form Normal-InvGamma posterior with either g-prior or ridge prior; posterior-predictive Monte Carlo at new points. Demo (n = 100, p = 3 covariates + intercept): with g = 100 the posterior mean matches OLS to two decimals and the 95% credible intervals cover all four true coefficients.
- `r/bayesian_linear_regression.R` — same closed-form update in base R. Production alternatives: `rstanarm::stan_glm`, `brms::brm`.

## When to use

- Small-to-moderate regression where you want honest posterior uncertainty and a proper posterior-predictive interval.
- Regularization framed probabilistically — ridge / lasso emerge as MAP under Normal / Laplace priors.
- Any downstream Bayesian decision analysis (loss functions on the posterior predictive).

## Assumptions

- Linear mean structure, homoscedastic Gaussian errors.
- Independent observations.
- Prior `m₀ = 0` on standardized covariates is usually harmless; shift/scale predictors before applying weakly informative priors.

## Beyond the conjugate case

- **Non-Gaussian errors** (Student-t, Laplace) → HMC / NUTS.
- **Sparse coefficients** → horseshoe or spike-and-slab priors → HMC or Gibbs with augmentation.
- **Hierarchical structure** → see `bayesian-hierarchical-models`.

## Run

```
python techniques/bayesian-linear-regression/python/bayesian_linear_regression.py
Rscript techniques/bayesian-linear-regression/r/bayesian_linear_regression.R
```

**Refs:** Zellner, A. *An Introduction to Bayesian Inference in Econometrics*, Wiley, 1971; Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., CRC, 2013 (Ch 14).

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

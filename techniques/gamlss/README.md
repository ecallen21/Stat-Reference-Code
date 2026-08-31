# GAMLSS — Location, Scale, Shape (Reference §33.6)

**Rigby & Stasinopoulos 2005.** Model the **full conditional
distribution** by letting every distribution parameter be its own
regression:

```
Y | X  ~  D(  μ  = f₁(X),
              σ  = f₂(X),
              ν  = f₃(X),   (skewness / df)
              τ  = f₄(X)    (kurtosis)   )
```

- **Gaussian**: `(μ, σ)` — heteroscedastic mean-variance regression.
- **Beta**: `(μ, φ)` — proportion regression with variable
  concentration.
- **Student-t**: `(μ, σ, ν)` — heteroscedastic with variable tail
  weight.
- **BCT, GB2, ZINB, …**: 100+ families in the R package.

## Advantages

- **Heteroscedasticity, skew, kurtosis are all regressed** — not just
  the mean.
- **Prediction intervals** correctly track scale.
- **Distribution-free family choice** — Gaussian, Gamma, Weibull,
  BCT, GB2, Beta, ZIP, ZINB, …

## When to use

- **Predictive distributions** matter (medical dosing, weather,
  financial risk).
- **Growth-curve reference charts** (WHO percentiles are GAMLSS).
- **Regression with clearly-variable variance / skewness**.

## When NOT to use

- **Very small n** — flexible families over-fit.
- **Point predictions only** — OLS / GLM is enough.
- **Extreme tails** — parametric families struggle; consider
  extreme-value distributions or quantile regression.

## Files

- `python/gamlss.py` — from-scratch Gaussian GAMLSS: `Y ~ N(X_μ β,
  exp(X_σ γ)²)`. Alternating gradient descent on `(β, γ)`. Demo on
  heteroscedastic data (true SD grows with `x`): **recovered
  β = [0.96, 0.47] (truth [1.0, 0.5]); log-sd γ = [-0.51, 0.60]
  (truth [-0.5, 0.6])**. Predictive-band width tracks true SD (0.96 /
  2.36 / 5.84 vs true 2z × [0.25, 0.61, 1.49]); OLS-fixed-sd is
  uniformly 3.42.
- `r/gamlss.R` — `gamlss` reference implementation, `brms` /
  `bamlss` for Bayesian variants; `pyGAM`, `ngboost`, `distfit`
  (Python).

## Assumptions & caveats

- **Family choice** — misspecified family biases location + variance
  jointly; pick with residual diagnostics (`wp`, worm plots).
- **Non-convex objective** — multiple restarts help.
- **Convergence** — mean and shape gradients on different scales;
  alternating IRLS or careful learning-rate choice.
- **Sample size** — recommend at least 10 obs per estimated distribution
  parameter × predictor.
- **Prediction intervals** — plug in `μ̂ ± z · σ̂(x)` for Gaussian;
  use the full quantile for skewed families.

## Related in this repo

- `distributional-regression` — a broader treatment.
- `quantile-regression`, `additive-quantile-regression`,
  `expectile-regression` — nonparametric alternatives.
- `bayesian-hierarchical-models`, `bayesian-glms` — Bayesian cousins.
- `beta-regression` — the fixed-shape special case for `[0, 1]` data.

## Run

```
python techniques/gamlss/python/gamlss.py
Rscript techniques/gamlss/r/gamlss.R
```

**Refs:** Rigby, R.A. & Stasinopoulos, D.M. "Generalized additive models for location, scale and shape." *Journal of the Royal Statistical Society, Series C*, 2005; Stasinopoulos, M. et al. *Flexible Regression and Smoothing: Using GAMLSS in R*, Chapman & Hall/CRC, 2017.

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

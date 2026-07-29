# Growth Curve Models (Reference §12.4)

Model individual trajectories over time with random intercepts and slopes:

```
y_{ij}  =  (β₀ + u_{0i})  +  (β₁ + u_{1i}) · t_{ij}  +  ε_{ij}
```

**Fixed effects** `(β₀, β₁)` = grand-average intercept and slope.
**Random effects** `(u_0, u_1) ~ N(0, G)` = subject-specific deviations from those averages.

Extends to **quadratic** growth by adding `β₂ · t²` (fixed) or `(β₂ + u_{2i}) · t²` (random).

Equivalent to a **latent growth model (LGM)** in SEM parameterization — the SEM formulation adds indirect / covariate paths but the core fit is the same LMM.

## What you get

- **Fixed effects**: population-average intercept + slope (+ quadratic).
- **G matrix**: variance of intercepts, variance of slopes, and their covariance.
- **BLUPs**: per-subject `(u_0, u_1)` — individual trajectories relative to the average.
- **Residual σ²**: within-subject noise.

## Files

- `python/growth_curve_models.py` — thin wrapper around [`linear-mixed-models`](../linear-mixed-models)'s `fit_lmm` with random intercept + random slope on time; linear and quadratic variants. Recovers β = [50.07, 2.01] vs. true [50, 2] on the demo.
- `r/growth_curve_models.R` — thin wrapper around `lme4::lmer(y ~ time + (time | subject))`.

## Assumptions

- Normal random effects and residuals.
- Correctly-specified functional form (linear / quadratic / spline).
- Missing data are MCAR or MAR (LMM handles automatically).

## Run

```
python techniques/growth-curve-models/python/growth_curve_models.py
Rscript techniques/growth-curve-models/r/growth_curve_models.R
```

**Refs:** Laird, N.M. & Ware, J.H. "Random-effects models for longitudinal data." *Biometrics* 38(4), 963–974, 1982; Singer, J.D. & Willett, J.B. *Applied Longitudinal Data Analysis: Modeling Change and Event Occurrence*, Oxford, 2003; Bollen, K.A. & Curran, P.J. *Latent Curve Models: A Structural Equation Perspective*, Wiley, 2006.

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

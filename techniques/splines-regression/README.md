# Spline Regression (Reference §5.12)

Approximate `f(x)` with **piecewise cubic polynomials** joined at knots so that the function, first derivative, and second derivative are all continuous.

## Three standard bases

- **Truncated-power cubic**: `(1, x, x², x³, (x − k_1)_+³, ..., (x − k_K)_+³)`. Simple; numerically ill-conditioned for many knots.
- **Natural cubic spline**: constrained linear beyond boundary knots. Better extrapolation, fewer degrees of freedom (`K` df with `K` interior knots + 2 boundary).
- **B-spline (de Boor)**: local support, numerically stable, standard for GAMs and interpolation. `scipy.interpolate.BSpline` / R's `splines::bs`.

Fit each by OLS on the basis matrix (unpenalized) or by ridge (penalized → smoothing splines / GAM).

## Files

- `python/splines_regression.py` — from-scratch truncated-power and natural bases + scipy-BSpline design matrix. Demo on n = 200, `sin(1.5x) + 0.3x + noise(0.3)`: all three bases give RMSE ≈ 0.07 vs truth with 6 interior knots; degree-6 polynomial reaches RMSE 0.055 with 6 df (marginally better on this smooth target).
- `r/splines_regression.R` — `splines::ns` and `splines::bs`.

## Choosing a basis

- **Natural cubic** — default for regression; safest at boundaries.
- **B-spline** — production choice; numerical stability + easy integration with penalty matrices.
- **Truncated-power** — pedagogically clearest; use for teaching, not fitting.

## Choosing knots

- **Quantile-based**: place knots at `linspace(0.1, 0.9, K)` quantiles of `x` — data-adaptive.
- **Equal-spaced**: place knots on an evenly-spaced grid over the range.
- **AIC / BIC / CV** search over knot count `K`.

Alternatively use a GAM (`techniques/gam`) with many knots + a smoothness penalty; the penalty picks the effective `K`.

## When to use

- **Nonlinear covariate effects** in a regression model where you don't need a parametric form.
- **Curve fitting** for signal recovery — dose-response, growth curves, physical measurements.
- **As a basis inside a GAM / GLM / Cox model** — natural cubic splines for `age`, `BMI`, `time` in Cox models.

## Assumptions & caveats

- **Boundary behavior**: truncated-power and B-splines can be wild at the edges; natural cubic constrains to linear.
- **Number of knots** dominates the bias-variance tradeoff; if you're not sure, use a GAM instead.
- **Colinearity**: many-knot bases produce highly correlated columns; solve numerically with QR or SVD rather than normal equations.

## Run

```
python techniques/splines-regression/python/splines_regression.py
Rscript techniques/splines-regression/r/splines_regression.R
```

**Refs:** Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch 5); de Boor, C. *A Practical Guide to Splines*, revised ed., Springer, 2001.

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

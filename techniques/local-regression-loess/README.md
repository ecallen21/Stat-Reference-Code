# Local Regression / LOESS (Reference §6.22)

At each target `x₀`, fit a low-degree polynomial (typically 1 or 2) to the data inside a neighborhood, weighted by distance to `x₀`:

`ŷ(x₀) = intercept of weighted local fit`, where `wᵢ = K((xᵢ − x₀)/h(x₀))`.

The **tricube** kernel is the default: `K(u) = (1 − |u|³)³` for `|u| < 1`. The bandwidth `h(x₀)` is chosen so the neighborhood holds a **span** fraction of the data — e.g. `span = 0.5` → half the data go into each local fit.

## Knobs

| Parameter | Meaning | Default |
|-----------|---------|---------|
| `span` | Fraction of points in each local neighborhood | 0.5 (R `loess`) / 0.75 (`lowess`) |
| `degree` | 1 (locally linear) or 2 (locally quadratic) | 2 |
| robust | Reweight outliers iteratively (LOWESS) | False (R `loess`) / True (`lowess`) |

Smaller `span` → wigglier (more local), bigger `span` → smoother. `degree = 2` handles curvature better near peaks; `degree = 1` is more robust.

## What LOESS is good for

- **Visual smoothing** of a scatterplot (e.g. `geom_smooth(method = "loess")` in ggplot2).
- **Trend isolation** for residual diagnostics — is there a curve in the residuals vs. fitted plot?
- **No-assumption baseline** before committing to a parametric form.

## What LOESS is not good for

- **Prediction far outside the data**. It's a local fit; extrapolation is meaningless.
- **High-dimensional `x`**. Implementations exist for 2-3 dimensions but the curse of dimensionality bites quickly.
- **Inference**. No clean asymptotic CI from a single LOESS fit; bootstrap if you need uncertainty.

## Files
- `python/local_regression_loess.py` — from-scratch tricube-weighted local polynomial fit (degree 1 or 2); demos span = 0.3 vs 0.7 to show the tradeoff; cross-checks against `statsmodels.nonparametric.lowess`.
- `r/local_regression_loess.R` — from-scratch + base `stats::loess` / `stats::lowess`.
- PySpark: N/A — at Spark scale, prefer parametric splines (`techniques/splines-segmented`) fit in MLlib, or sample down before running LOESS.

## Run
```
python techniques/local-regression-loess/python/local_regression_loess.py
Rscript techniques/local-regression-loess/r/local_regression_loess.R
```

**Refs:** Cleveland, "Robust Locally Weighted Regression and Smoothing Scatterplots," *JASA* 74(368), 829–836, 1979; Cleveland & Devlin, "Locally Weighted Regression: An Approach to Regression Analysis by Local Fitting," *JASA* 83(403), 596–610, 1988.

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

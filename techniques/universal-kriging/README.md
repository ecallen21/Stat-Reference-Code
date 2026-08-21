# Universal (Drift) Kriging (Reference §23.x extra)

Extends ordinary kriging to non-constant mean:

```
Z(s) = X(s) β + δ(s)
```

- `X(s)` is a `p`-column **drift** design (polynomial in coordinates, elevation, land-use dummies, …).
- `δ(s)` is a zero-mean stationary residual with a known / estimated **variogram**.

## Augmented kriging system

Predict at `s₀`; find weights `λ` and Lagrange multipliers `μ`:

```
[ Σ    X ]   [ λ ]   [ σ₀ ]
[ Xᵀ   0 ]   [ μ ] = [ x₀ ]
```

- `Σ_ij = cov(δ_i, δ_j)`.
- `σ₀_i = cov(δ_i, δ(s₀))`.
- `x₀ = X(s₀)` — drift vector at the target.

Prediction: `Ẑ(s₀) = λᵀ Z`; variance: `σ² − λᵀ σ₀ − μᵀ x₀`.

## Special cases

| Setup | Name |
|---|---|
| `X = 1` (constant) | **Ordinary kriging** (see `ordinary-kriging`) |
| `X = [1, x, y]` (linear trend) | **Universal kriging** (this module) |
| `X = [1, elev(s)]` (single covariate) | **Kriging with external drift** (KED) |
| `X = [1, x, y, x², xy, y²]` (quadratic trend) | Higher-order UK |
| Nonlinear regression + kriged residuals | **Regression kriging** (equivalent under exact-fit assumptions) |

## When to use

- **Broad-scale trend + local variation** — elevation gradients, north-south pollution transects, urban-rural income gradients.
- **A covariate is measured densely** (satellite imagery, DEM) and can be used as `X(s)` to inform predictions.
- **Physically motivated deterministic component** — e.g. distance-to-coast for salinity.

## Files

- `python/universal_kriging.py` — from-scratch UK with an exponential covariance and a `[1, x, y]` drift design. Demo (n=60 observations, linear NS+EW trend + exponential-covariance residual, sill 1, range 2): LOO RMSE 0.65 with true residual SD 1 and noise SD 0.2; predicted range −0.40…4.96 vs observed −0.89…5.12; mean prediction variance 0.38.
- `r/universal_kriging.R` — `gstat::krige(z ~ x + y, ...)`, `automap::autoKrige`, `fields::mKrig`.

## Assumptions & caveats

- **Variogram of the residuals**, not of `Z` — estimate the variogram after removing the trend (iteratively: fit β by OLS, model the residual variogram, refit by GLS, iterate). `gstat::variogram(z ~ x + y, ...)` does this automatically.
- **Extrapolation is dangerous** — the drift extrapolates beyond the observed covariate range; the residual kriging cannot rescue a badly extrapolated trend.
- **Colinearity between drift columns** blows up the augmented system; regularise with a small ridge on the top-left block.
- **KED requires the covariate at every prediction location** — pixel-aligned DEM or satellite scene.
- **Design matrix inclusion** breaks strict stationarity of `Z` but keeps stationarity of `δ`; report both trend coefficients and variogram parameters.

## Run

```
python techniques/universal-kriging/python/universal_kriging.py
Rscript techniques/universal-kriging/r/universal_kriging.R
```

**Refs:** Matheron, G. *La Théorie des Variables Régionalisées et ses Applications*, École des Mines, 1971; Cressie, N. *Statistics for Spatial Data*, Wiley, 1993; Chilès, J.-P. & Delfiner, P. *Geostatistics: Modeling Spatial Uncertainty*, 2nd ed., Wiley, 2012.

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

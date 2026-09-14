# Thin-Plate Splines (Reference §47.328)

Duchon (1977); Wahba (1990). Minimum-curvature smoothing of
2-D scattered data:

```
min Σ (f(x_i) − y_i)² + λ ∫ (f_xx² + 2 f_xy² + f_yy²) dx dy
```

The optimum has the form

```
f(x) = α_0 + α_1 x_1 + α_2 x_2 + Σ w_i K(‖x − x_i‖)
K(r) = r² log(r)
```

Closed-form solution via a block linear system. Standard for
2-D interpolation and geostatistics (kriging with linear
drift).

## Files

- `python/thin_plate_splines.py` — Direct TPS fit on 60
  scattered points from `z = sin(2πx) + 0.5 cos(4πy)`. Grid
  RMSE ≈ 0.3 with λ=0.01. Corner predictions demonstrate
  the trade-off: TPS extrapolates smoothly but loses accuracy
  where data are sparse.
- `r/thin_plate_splines.R` — `fields::Tps`, `mgcv::gam(bs='tp')`
  (R); `scipy.interpolate.RBFInterpolator('thin_plate_spline')`,
  from-scratch (Python).

## When to use

- **2-D smoothing** of scattered data — geospatial,
  environmental, biomedical.
- **GAM smoother** — `mgcv` uses TPS as the default penalised
  smoother.
- **Warping / image registration** — TPS as parametric warp.

## When NOT to use

- **Large N (> ~5000)** — direct TPS is O(N³); use
  low-rank / sparse TPS approximations (Wood's tprs).
- **Discontinuous surfaces** — TPS smooths through edges;
  use piecewise / kernel methods.
- **Higher-dimensional data (> 3)** — kernel changes; use
  Matérn / Gaussian RBF instead.

## Assumptions & caveats

- **λ (smoothing)** — controls fit vs smoothness; select
  via GCV / REML.
- **Boundary conditions** — TPS grows unboundedly outside
  data hull; extrapolation is dangerous.
- **Reduced-rank TPS** (tprs, Wood 2003) scales to large N.
- **Anisotropic data** — pre-scale coordinates so that
  variability matches.

## Related in this repo

- `splines-regression`, `functional-basis-smoothing`,
  `gamlss` — related smoothing tools.
- `gaussian-process-regression`, `sparse-gaussian-process`,
  `ordinary-kriging`, `geographically-weighted-regression`
  — spatial-statistics alternatives.
- `kernel-density-estimation` — related nonparametric
  smoother.

## Run

```
python techniques/thin-plate-splines/python/thin_plate_splines.py
Rscript techniques/thin-plate-splines/r/thin_plate_splines.R
```

**Refs:** Duchon, J. "Splines minimizing rotation-invariant semi-norms in Sobolev spaces." In *Constructive Theory of Functions of Several Variables*, Springer, 1977; Wahba, G. *Spline Models for Observational Data*, SIAM, 1990.

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

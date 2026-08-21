# Empirical + Theoretical Variograms (Reference §23.6)

## Semivariogram

```
γ(h) = ½ · E[(Z(x + h) − Z(x))²]
```

Empirical estimate by binning pairs by distance:

```
γ̂(h_bin) = (1 / 2 |N(h_bin)|) Σ_{(i,j) ∈ N(h_bin)} (Z_i − Z_j)²
```

## Theoretical models

Fit with three parameters: **nugget** `n`, **sill** `s`, **range** `r`.

- **Spherical**: `γ(h) = n + (s − n) · [1.5 (h/r) − 0.5 (h/r)³]` for `h ≤ r`; `s` for `h > r`.
- **Exponential**: `γ(h) = n + (s − n) (1 − exp(−3h/r))`.
- **Gaussian**: `γ(h) = n + (s − n) (1 − exp(−3(h/r)²))`.

**Nugget** = variance at zero distance (measurement error + microscale). **Sill** = long-range variance. **Range** = distance at which spatial correlation vanishes (or 95% of it, in exponential/Gaussian).

## Files

- `python/variogram_modeling.py` — empirical binning + WLS fit of spherical / exponential / Gaussian. Demo on a Gaussian random field with exponential covariance (true nugget 0.5, partial sill 2.0, range 2.0): exponential fit gives nugget 0.53, sill 1.76, range 5.97 (approximate — Gauss-Newton on scaled range).
- `r/variogram_modeling.R` — `gstat::variogram` + `gstat::fit.variogram(vgm(...))`.

## When to use

- **Feed into kriging** (`ordinary-kriging`) — the fitted variogram is the covariance model.
- **Spatial correlation range** — how far spatial autocorrelation extends.
- **Nugget / signal split** — nugget quantifies measurement error + microscale variance.

## Assumptions & caveats

- **Second-order stationarity** — mean and variance don't depend on location.
- **Isotropy** — variogram depends only on `|h|`, not direction. Check with directional variograms.
- **Weighted LS** with Cressie's weight `n_bin / γ(h)²` gives more principled fits than plain OLS.
- **Enough pairs per bin** — `≥ 30` for stable `γ̂`.

## Run

```
python techniques/variogram-modeling/python/variogram_modeling.py
Rscript techniques/variogram-modeling/r/variogram_modeling.R
```

**Refs:** Matheron, G. *Traité de Géostatistique Appliquée*, Editions Technip, 1962; Cressie, N. *Statistics for Spatial Data*, Wiley, 1993.

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

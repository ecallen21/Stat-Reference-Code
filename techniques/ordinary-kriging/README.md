# Ordinary Kriging (Reference §23.7)

Best Linear Unbiased Predictor (BLUP) of `Z(x_0)` at an unsampled location:

```
Ẑ(x_0) = Σ_i λ_i Z(x_i)          subject to  Σ_i λ_i = 1  (unbiasedness)
```

Weights minimize prediction variance → solve the kriging linear system:

```
| Γ   1 | | λ |    | γ_0 |
| 1ᵀ  0 | | μ |  = |  1  |
```

where `Γ_ij = γ(‖x_i − x_j‖)` and `γ_0[i] = γ(‖x_i − x_0‖)` from a fitted variogram (see `variogram-modeling`).

Kriging variance:

```
σ²_K(x_0) = λᵀ γ_0 + μ
```

## Files

- `python/ordinary_kriging.py` — from-scratch kriging system solver with an exponential variogram. Demo (n = 80 samples, `sin(x) + cos(y)` field): RMSE on 5×5 grid = 0.46 (noise SD 0.3 in training data).
- `r/ordinary_kriging.R` — `gstat::krige`.

## When to use

- **Spatial interpolation** — geological / environmental / meteorological variables (temperature, rainfall, soil, ore).
- **Provides uncertainty** — kriging variance is a rigorous prediction SE, unlike IDW.
- **Optimal under stationarity + Gaussianity** — best you can do without a parametric mean.

## Variants

- **Simple kriging** — known mean; no unbiasedness constraint.
- **Universal kriging / Kriging with external drift** — model mean as a linear function of covariates.
- **Co-kriging** — jointly predict multiple correlated variables.
- **Indicator kriging** — for probabilities of exceeding thresholds.
- **Block kriging** — predict areal averages instead of points.

## Assumptions & caveats

- **Second-order stationarity** — mean constant, variance stationary; check by removing trends first.
- **Correct variogram** critical; sensitivity to variogram model + parameters.
- **Cost** `O(n³)` per prediction (solve system per new point); use local kriging with the `k` nearest samples for scale.
- **Extrapolation** outside the sample envelope reverts toward the mean.

## Run

```
python techniques/ordinary-kriging/python/ordinary_kriging.py
Rscript techniques/ordinary-kriging/r/ordinary_kriging.R
```

**Refs:** Krige, D.G. "A statistical approach to some basic mine valuation problems on the Witwatersrand." *J. Chem. Metall. Min. Soc. S. Afr.* 52, 119–139, 1951; Cressie, N. *Statistics for Spatial Data*, Wiley, 1993; Chilès, J.-P. & Delfiner, P. *Geostatistics: Modeling Spatial Uncertainty*, 2nd ed., Wiley, 2012.

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

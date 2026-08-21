# Spatial Autoregressive Models (Reference §23.9)

## Spatial lag (SAR-lag / SLM)

```
y = ρ · W y + X β + ε           ε ~ N(0, σ² I)
```

`ρ` measures spatial dependence in the **outcome** — spillover of neighbours' `y` into own `y`.

## Spatial error (SAR-error / SEM)

```
y = X β + u
u = λ · W u + ε
```

`λ` measures spatial dependence in the **error** — omitted-variable spatial structure.

## Estimation (concentrated MLE)

Profile out `(β, σ²)` and search over `ρ` (or `λ`):

```
log-lik(ρ) = log|I − ρW| − (n/2) log(2π σ̂²(ρ)) − n/2
```

`σ̂²(ρ)` and `β̂(ρ)` computed at each ρ. Search over the stability region `(1/λ_min(W), 1/λ_max(W))`.

## Files

- `python/spatial_autoregressive_sar.py` — from-scratch SAR lag and SAR error concentrated MLE with grid-then-refine ρ search. Demo (100-node grid, true ρ = 0.5, spatially correlated x): SAR-lag ρ̂ ≈ 0.31 (attenuated); OLS β_x = 0.94 badly biased vs true 0.5. Attenuation illustrates the price of a small n and coarse grid.
- `r/spatial_autoregressive_sar.R` — pointers to `spatialreg::lagsarlm`, `spatialreg::errorsarlm`, `spatialreg::sacsarlm`.

## When to use

- **SAR-lag** — outcome literally depends on neighbours' outcomes (real-estate prices, disease diffusion).
- **SAR-error** — spatial structure in unobserved covariates (soil, climate) creates residual spatial correlation.
- **LM tests** (Anselin 1988) pick between lag and error based on residual diagnostics.

## Assumptions & caveats

- **Choice of W** matters enormously — sensitivity analyses across contiguity / kNN / kernel.
- **Identifiability** — with row-standardized W and spatially uncorrelated regressors, spatial signal averages away.
- **Bounded ρ / λ** — stability requires `(1 − ρW)` to be non-singular; row-standardized W bounds ρ in (−1, 1).
- **Prefer full likelihood over IV** for small samples; IV alternatives exist for very large n.

## Run

```
python techniques/spatial-autoregressive-sar/python/spatial_autoregressive_sar.py
Rscript techniques/spatial-autoregressive-sar/r/spatial_autoregressive_sar.R
```

**Refs:** Anselin, L. *Spatial Econometrics: Methods and Models*, Kluwer, 1988; LeSage, J.P. & Pace, R.K. *Introduction to Spatial Econometrics*, CRC, 2009.

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

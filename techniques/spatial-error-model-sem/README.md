# Spatial Error Model (SEM) (Reference §47.48)

Anselin (1988). Companion to the spatial autoregressive (SAR) lag
model. SEM places spatial correlation in the **disturbances**
rather than the outcome:

    y = X β + u,     u = λ W u + ε,     ε ~ N(0, σ² I).

Point estimates of β remain interpretable as usual regression
effects; the λ term corrects standard errors and Wald tests.
Ignoring spatial residual dependence biases inference (not point
estimates). Fitted here by profile likelihood over λ.

## Files

- `python/spatial_error_model_sem.py` — from-scratch profile-
  likelihood MLE with row-normalised queen-contiguity W on a grid.
  Demo (12×12 grid, true λ=0.6, β=(1, 2, −0.5)):
  - Naive OLS β = (0.96, 2.01, −0.44)
  - SEM MLE  β = (0.96, 1.98, −0.46), λ̂ = 0.42.
- `r/spatial_error_model_sem.R` — `spatialreg::errorsarlm`,
  `spdep::lm.LMtests` (R); `pysal.model.spreg.GM_Error`,
  from-scratch (Python).

## When to use

- **Areal-unit regression** (counties, census tracts) with
  spatially clustered omitted variables.
- **After Moran's I on residuals** flags spatial dependence.
- **When theory says shocks (not means) diffuse** across units.
- **Robust-inference alternative to spatial HAC** with small N.

## When NOT to use

- **True lag of y matters** — dependent variable in y_i depends on
  y_j — use SAR-lag (`spatial-autoregressive-sar`).
- **Both lag and error** — use SARAR / SAC model.
- **Very sparse or fragmented W** — profile likelihood surface flat;
  large SE on λ.
- **Non-Gaussian errors** — extend to GLM-SEM or use bootstrap SEs.

## Assumptions & caveats

- **W specification** — queen vs rook vs k-nearest; results can be
  sensitive to spatial neighbours choice.
- **|λ| < 1** for stationarity (or 1/eigmax(W) bound).
- **Homoscedastic ε** — extend to heteroskedastic SEM (Kelejian-
  Prucha 2010) if needed.
- **Identifiability of β from λ** — beware SLX (spatial X) plus SEM;
  see Elhorst 2010 taxonomy.

## Related in this repo

- `spatial-autoregressive-sar` — SAR-lag counterpart.
- `spatial-diff-in-diff`, `spatial-glm`, `spatial-weights-matrix`,
  `spatial-scan-cluster` — spatial toolkit.
- `newey-west-hac` — HAC standard errors (temporal analogue).
- `geographically-weighted-regression` — spatially varying
  coefficients.

## Run

```
python techniques/spatial-error-model-sem/python/spatial_error_model_sem.py
Rscript techniques/spatial-error-model-sem/r/spatial_error_model_sem.R
```

**Refs:** Anselin, L. *Spatial Econometrics: Methods and Models.* Kluwer Academic Publishers, Dordrecht, 1988; Elhorst, J.P. "Applied spatial econometrics: raising the bar." *Spatial Econ Analysis* 5(1): 9-28, 2010.

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

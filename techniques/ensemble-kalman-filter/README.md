# Ensemble Kalman Filter (EnKF) (Reference §18.11)

Evensen (1994). Monte-Carlo Kalman filter: represent the posterior
by an ensemble of state vectors, propagate them through nonlinear
dynamics, and use empirical ensemble covariance in the Kalman
update.

## Cycle

1. **Forecast**: `x_iᶠ = f(x_iᵃ,t−1) + η_i`.
2. Empirical `x̄ᶠ`, `Pᶠ` from the ensemble.
3. Perturb observations: `y_i = y + ε_i`.
4. Kalman gain: `K = Pᶠ Hᵀ (H Pᶠ Hᵀ + R)⁻¹`.
5. **Update**: `x_iᵃ = x_iᶠ + K (y_i − H x_iᶠ)`.

## Files

- `python/ensemble_kalman_filter.py` — full EnKF cycle from scratch
  on a 2-D nonlinear model `x_{t+1} = 0.9x + 0.4 sin(x) + noise`.
  Demo (T=60, N=40): filter-mean RMSE 0.463 vs observation RMSE
  1.058 — ensemble averaging shrinks noise.
- `r/ensemble_kalman_filter.R` — `dartR`, `MFEnKF`, `KFAS` (R);
  `filterpy.kalman.EnsembleKalmanFilter`, `dapper` (Python).

## When to use

- **High-dim geophysical / atmospheric models** — weather, ocean,
  reservoir data assimilation.
- **Nonlinear dynamics** — no need to linearise (unlike EKF).
- **Uncertainty quantification** — ensemble provides posterior
  samples directly.

## When NOT to use

- **Linear-Gaussian problems** — plain Kalman is optimal and
  cheaper.
- **Very small N** — sampling error inflates covariance; use
  localisation / inflation.
- **Multimodal posteriors** — Gaussian assumption of the update
  fails; use particle filter.

## Assumptions & caveats

- **Gaussian update approximation** — only the update step assumes
  Gaussianity; forecast is exact under `f`.
- **Localisation** — spatial correlation cutoff to combat spurious
  long-range covariances at small N.
- **Inflation** — multiply ensemble spread by 1.02–1.10 to counter
  under-dispersion.
- **Perturbed-observation form** shown here vs deterministic
  square-root filter (ETKF); latter is common in operational NWP.

## Related in this repo

- `state-space-kalman`, `state-space-models`, `extended-kalman-filter`,
  `unscented-kalman-filter`, `particle-filter-smc` — filtering
  siblings.
- `mcmc-metropolis-hastings`, `variational-inference` — Bayesian
  alternatives for state inference.

## Run

```
python techniques/ensemble-kalman-filter/python/ensemble_kalman_filter.py
Rscript techniques/ensemble-kalman-filter/r/ensemble_kalman_filter.R
```

**Refs:** Evensen, G. "Sequential data assimilation with a nonlinear quasi-geostrophic model using Monte Carlo methods to forecast error statistics." *JGR*, 99(C5): 10143-10162, 1994; Evensen, G. "The ensemble Kalman filter: theoretical formulation and practical implementation." *Ocean Dynamics*, 53(4): 343-367, 2003.

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

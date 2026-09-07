# Unscented Kalman Filter (UKF) (Reference §18.13)

Julier & Uhlmann (1997). Replaces the EKF's linearisation with
**deterministic sigma points** that capture the mean and covariance
of the state, propagated through the nonlinear `f`, `h` and
recombined to produce the transformed mean / covariance.

## Sigma points

    χ₀ = m
    χ_i = m + √((d + λ) P)_i        i = 1..d
    χ_i = m − √((d + λ) P)_{i−d}    i = d+1..2d

with tuning `α, β, κ` (defaults α=1e-3, β=2, κ=0). Propagate `Y_i =
f(χ_i)`, then reweight with `w_m`, `w_c`.

## Files

- `python/unscented_kalman_filter.py` — sigma-point + UKF cycle
  from scratch. Demo (same 1-D nonlinear/quadratic problem as EKF,
  T=80): UKF RMSE 0.265 ≈ EKF 0.264 for this mild nonlinearity.
  UKF pulls ahead when curvature is stronger.
- `r/unscented_kalman_filter.R` — `mvKf`, `ukf`, `bssm` (R);
  `filterpy.kalman.UnscentedKalmanFilter`, `pyfilter`, `dapper`
  (Python).

## When to use

- **Strong non-linearity** — where EKF's linearisation breaks
  down.
- **No analytic Jacobians** — UKF is derivative-free.
- **Discontinuous transformations** — UKF handles them; EKF cannot.
- **Real-time systems** with tight compute — UKF is deterministic
  (2d+1 propagations) vs particle filter's variance.

## When NOT to use

- **Highly Gaussian problems** — plain Kalman is optimal.
- **Multimodal posteriors** — sigma points only capture mean/cov;
  use particle filter.
- **Very high-dim state** — 2d+1 propagations become expensive;
  EnKF is preferred.

## Assumptions & caveats

- **Gaussian approximation** — UKF still assumes the posterior is
  well-summarised by mean and covariance.
- **Tuning α, β, κ** — small α centres sigma points; β=2 is optimal
  for Gaussians.
- **Numerical positive-definiteness** — Cholesky decomposition of
  `(d + λ)P` needs P PSD; use Cholesky-updates or square-root UKF
  for stability.
- **Higher-order UKF** — cubature KF (CKF) uses spherical-radial
  points for exact 3rd-order Gaussian integrals.

## Related in this repo

- `state-space-kalman`, `state-space-models`,
  `extended-kalman-filter`, `ensemble-kalman-filter`,
  `particle-filter-smc` — filtering siblings.
- `nonlinear-least-squares`, `neural-ode` — nonlinear-dynamics
  cousins.

## Run

```
python techniques/unscented-kalman-filter/python/unscented_kalman_filter.py
Rscript techniques/unscented-kalman-filter/r/unscented_kalman_filter.R
```

**Refs:** Julier, S.J. & Uhlmann, J.K. "A new extension of the Kalman filter to nonlinear systems." *SPIE AeroSense*, 3068, 1997; Wan, E.A. & van der Merwe, R. "The unscented Kalman filter for nonlinear estimation." *IEEE AS-SPCC*, 2000.

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

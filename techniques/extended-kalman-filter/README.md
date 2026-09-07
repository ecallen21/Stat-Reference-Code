# Extended Kalman Filter (EKF) (Reference §18.12)

Kalman filter extended to nonlinear dynamics / observations by
**first-order linearisation** around the current mean:

    x_{t+1} = f(x_t) + η_t
    y_t     = h(x_t) + ε_t

Each cycle replaces `f`, `h` with their Jacobians evaluated at the
current estimate.

## Cycle

    Predict:  x = f(x_prev),          P = F P Fᵀ + Q
    Update:   K = P Hᵀ (H P Hᵀ + R)⁻¹
              x = x + K (y − h(x))
              P = (I − K H) P

## Files

- `python/extended_kalman_filter.py` — general EKF from scratch on
  a 1-D nonlinear state / quadratic observation system. Demo
  (T=80): EKF RMSE vs truth 0.264.
- `r/extended_kalman_filter.R` — `KFAS + numDeriv::jacobian`,
  `FKF`, `bssm` (R); `filterpy.kalman.ExtendedKalmanFilter`
  (Python).

## When to use

- **Mild non-linearity** — GPS/INS navigation, robotics, target
  tracking, small-angle mechanics.
- **Well-defined analytic Jacobians** — automatic differentiation
  (jax, torch) helps.
- **Low-dimensional state** where linearisation error is small.

## When NOT to use

- **Strong nonlinearity / bimodality** — UKF, particle filter, or
  EnKF are more robust.
- **Discontinuous dynamics** — Jacobian undefined at jumps.
- **When Jacobians are unstable** — use derivative-free (UKF)
  alternatives.

## Assumptions & caveats

- **First-order Taylor** — biased when second-order effects
  dominate; higher-order EKFs exist but are rarely used.
- **Positive-definite covariance** — numerical roundoff can lose
  it; use Joseph form or square-root filter.
- **Filter divergence** — a bad linearisation propagates; monitor
  innovation z-scores and inflate `Q` if needed.
- **Initial conditions matter** — poor `x₀`, `P₀` slow convergence.

## Related in this repo

- `state-space-kalman`, `state-space-models`,
  `ensemble-kalman-filter`, `unscented-kalman-filter`,
  `particle-filter-smc` — filtering siblings.
- `nonlinear-least-squares`, `neural-ode` — nonlinear-dynamics
  cousins.

## Run

```
python techniques/extended-kalman-filter/python/extended_kalman_filter.py
Rscript techniques/extended-kalman-filter/r/extended_kalman_filter.R
```

**Refs:** Jazwinski, A.H. *Stochastic Processes and Filtering Theory*, Academic Press, 1970; Anderson, B.D.O. & Moore, J.B. *Optimal Filtering*, Prentice-Hall, 1979; Simon, D. *Optimal State Estimation: Kalman, H-Infinity, and Nonlinear Approaches*, Wiley, 2006.

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

# Particle Filter / Sequential Monte Carlo (Reference §18.10)

Gordon, Salmond & Smith (1993). Filtering for a nonlinear /
non-Gaussian state-space model

    x_t = f(x_{t−1}) + η_t
    y_t = g(x_t) + ε_t

by a weighted particle approximation of `p(x_t | y_{1:t})`.

## Bootstrap-filter (SIR)

1. Initialise `K` particles `x_0^(k) ~ p(x_0)`.
2. For `t = 1, …, T`:
   - **Propagate**: `x_t^(k) = f(x_{t−1}^(k)) + η_t^(k)`
   - **Weight**: `w_t^(k) ∝ p(y_t | x_t^(k))`
   - **Resample** `x_t^(k)` with probability `w_t^(k)`

Effective sample size `ESS = 1 / Σ w²` monitors particle degeneracy;
systematic resampling when ESS < threshold (often K/2).

## Files

- `python/particle_filter_smc.py` — bootstrap SIR filter with
  systematic resampling from scratch, plus a Kalman filter for the
  linear-Gaussian sanity check. Demo (T=80, AR(1) state, Gaussian
  obs): PF (K=800) RMSE 0.582 ≈ Kalman RMSE 0.583. Nonlinear obs
  `y = |x| + ε`: PF RMSE 1.13 (Kalman needs EKF/UKF here).
- `r/particle_filter_smc.R` — `pomp::pfilter`, `nimbleSMC`, `SMC`,
  `KFAS::simSSM` (R); `particles`, `pyfilter`, `pymc` (Python).

## When to use

- **Nonlinear or non-Gaussian SSMs** — where Kalman/EKF fail
  (bearings-only tracking, ecological population dynamics, epidemic
  transmission).
- **Multimodal filtering distributions** — Gaussian filters can't
  represent multiple hypotheses; particles can.
- **Online / streaming** — the filter is causal and updates one
  step at a time.
- **Likelihood estimation for PMCMC** — the average of unnormalised
  weights is an unbiased estimator of `p(y_{1:t})`, used inside
  particle-MCMC.

## When NOT to use

- **Linear-Gaussian SSMs** — Kalman is optimal and free of Monte-Carlo
  noise.
- **High-dimensional state (curse of dimensionality)** — vanilla PF
  suffers weight collapse; use annealed / auxiliary / block PF.
- **Very small K** — you need enough particles to cover the
  posterior support; ESS collapses fast.

## Assumptions & caveats

- **Observation likelihood `p(y | x)`** must be evaluable up to
  proportionality.
- **Resampling variance** — degrades the ESS; use systematic /
  stratified resampling rather than multinomial.
- **Path degeneracy** — trajectories collapse to few ancestors at
  older times; use fixed-lag smoothing.
- **Bias-vs-variance tuning** — auxiliary and adapted proposals cut
  variance at the cost of extra design.

## Related in this repo

- `state-space-kalman`, `state-space-models` — linear Gaussian
  filtering / smoothing.
- `mcmc-metropolis-hastings`, `hmc-nuts`, `variational-inference`
  — Bayesian inference alternatives.
- `hmm` — discrete-state cousin with exact forward-backward.
- `abc-approximate-bayesian` — likelihood-free counterpart when
  even `p(y|x)` is intractable.

## Run

```
python techniques/particle-filter-smc/python/particle_filter_smc.py
Rscript techniques/particle-filter-smc/r/particle_filter_smc.R
```

**Refs:** Gordon, N.J., Salmond, D.J. & Smith, A.F.M. "Novel approach to nonlinear/non-Gaussian Bayesian state estimation." *IEE Proceedings F*, 140(2): 107-113, 1993; Doucet, A., de Freitas, N. & Gordon, N. *Sequential Monte Carlo Methods in Practice*, Springer, 2001.

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

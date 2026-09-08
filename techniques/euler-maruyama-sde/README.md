# Euler-Maruyama SDE Simulation (Reference §47.57)

Maruyama (1955). Numerical integrator for the Itô SDE
`dX_t = μ(X_t, t) dt + σ(X_t, t) dW_t`:

    X_{k+1} = X_k + μ(X_k, t_k) Δt + σ(X_k, t_k) √Δt · Z_k,   Z_k ~ N(0, 1).

Strong order 0.5, weak order 1. Milstein scheme adds an Itô
correction to reach strong order 1; SRK / Runge-Kutta variants
push further at extra cost.

## Files

- `python/euler_maruyama_sde.py` — generic EM integrator +
  demos. Geometric Brownian motion (r=0.05, σ=0.3, S₀=100, T=1,
  10 000 paths):
  - E[S_T] MC = 104.84 vs analytic 105.13
  - Var[S_T] MC = 1037 vs analytic 1041.
  Ornstein-Uhlenbeck (κ=3, θ=1, η=0.4):
  - stationary mean 1.0009 vs 1.0; stationary var 0.0268 vs 0.0267.
- `r/euler_maruyama_sde.R` — `Sim.DiffProc`, `yuima`,
  `sde` (R); `sdeint`, `diffrax`, from-scratch (Python).

## When to use

- **Quantitative finance** — GBM for Black-Scholes; Heston,
  SABR, CIR for rates & vol.
- **Population dynamics** — Wright-Fisher, logistic-noise SDEs.
- **Physics** — Langevin dynamics, Brownian motors.
- **MCMC** — Langevin (MALA) samplers use an EM step.

## When NOT to use

- **Path-dependent derivatives with tight discretisation error
  bounds** — use Milstein (order 1) or higher.
- **Stiff SDEs** — implicit Euler-Maruyama or IRK schemes.
- **Jumps** — need EM extensions for Lévy / jump-diffusion.
- **Very small dt with large # paths** — vectorise across paths,
  not steps.

## Assumptions & caveats

- **Δt small enough** — bias scales O(Δt); halve to check.
- **Strong vs weak convergence** — moments (weak) converge faster
  than paths (strong).
- **Stability** — geometric BM may go negative under EM at moderate
  Δt; log-transform first (dY = (r − σ²/2) dt + σ dW).
- **Random-number reuse** — use antithetic + common random numbers
  for variance reduction across sensitivities.

## Related in this repo

- `monte-carlo-simulation`, `importance-sampling`,
  `quasi-monte-carlo-sobol`, `antithetic-control-variates` — MC
  toolkit.
- `hmc-nuts`, `hamiltonian-mc`, `stochastic-gradient-mcmc` —
  Hamiltonian / Langevin-flavour MCMC.
- `state-space-kalman`, `particle-filter-smc`, `neural-ode`,
  `state-space-models` — continuous-time state models.
- `diffusion-model`, `score-matching` — generative modelling with
  reverse-time SDEs.

## Run

```
python techniques/euler-maruyama-sde/python/euler_maruyama_sde.py
Rscript techniques/euler-maruyama-sde/r/euler_maruyama_sde.R
```

**Refs:** Maruyama, G. "Continuous Markov processes and stochastic equations." *Rend Circ Mat Palermo* 4: 48-90, 1955; Kloeden, P.E. & Platen, E. *Numerical Solution of Stochastic Differential Equations.* Springer, 1992.

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

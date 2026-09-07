# Adaptive Metropolis (Haario 2001) (Reference §25.8)

Haario, Saksman & Tamminen (2001). Random-walk Metropolis-Hastings
whose **proposal covariance is updated** during sampling to match
the empirical posterior covariance:

    Σ_t = (2.38² / d) · Cov(X_1, …, X_{t−1})  +  ε · I

Diminishing-adaptation and containment conditions (Roberts &
Rosenthal 2007) ensure the chain remains ergodic despite the
non-Markov proposals.

## Files

- `python/adaptive_metropolis_haario.py` — naive RWMH vs adaptive
  Metropolis with running-mean/cov (Welford) from scratch. Demo
  (2-D anisotropic Gaussian, ρ=0.9): both recover the covariance
  correctly, but AM cuts lag-1 autocorrelation from 0.991 (RWMH)
  to 0.756 — the chain mixes far better once the proposal aligns
  with the target.
- `r/adaptive_metropolis_haario.R` — `adaptMCMC`, `MCMCpack`,
  `BayesianTools`, `mcmc::metrop` (R); pymc AdaptiveMetropolis,
  emcee, from-scratch (Python).

## When to use

- **Correlated / poorly-scaled targets** — hierarchical models with
  varying-scale parameters.
- **No time to hand-tune step size** — production Bayesian
  workflows.
- **Moderate dimensions** — d up to a few hundred; higher-d needs
  HMC / NUTS.

## When NOT to use

- **Very high dimensions** or **strongly multimodal targets** — HMC
  / NUTS / SMC alternatives.
- **Very short chains** — the covariance estimate needs enough
  samples to be useful.

## Assumptions & caveats

- **Diminishing adaptation** — the adaptation rate must decay
  (implicit in Welford averaging) so the chain is still ergodic.
- **Warm-up window** — start with a small fixed proposal; only
  adapt after ~200-500 samples.
- **Numerical safeguard** — `+ε · I` in the proposal cov keeps it
  positive definite when early samples are degenerate.
- **Optimal acceptance** — target 0.234 for high-d Gaussian (Roberts-
  Gelman-Gilks 1997); AM's constant `2.38²/d` implements this.

## Related in this repo

- `mcmc-metropolis-hastings`, `gibbs-sampler`, `hmc-nuts` — MCMC
  siblings.
- `hamiltonian-mc`, `variational-inference` — alternative posterior
  approximators.
- `stochastic-gradient-mcmc` — the mini-batch cousin.

## Run

```
python techniques/adaptive-metropolis-haario/python/adaptive_metropolis_haario.py
Rscript techniques/adaptive-metropolis-haario/r/adaptive_metropolis_haario.R
```

**Refs:** Haario, H., Saksman, E. & Tamminen, J. "An adaptive Metropolis algorithm." *Bernoulli*, 7(2): 223-242, 2001; Roberts, G.O. & Rosenthal, J.S. "Coupling and ergodicity of adaptive Markov chain Monte Carlo algorithms." *Journal of Applied Probability*, 44: 458-475, 2007.

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

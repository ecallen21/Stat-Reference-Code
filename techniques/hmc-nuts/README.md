# HMC / NUTS (Reference §14.4)

Duane et al. (1987), Neal (2011), Hoffman & Gelman (2014).
Hamiltonian Monte Carlo simulates physics-inspired dynamics to
propose distant states along the target posterior:

- Position `q` = parameter, momentum `p` refreshed from
  `N(0, M)` each iteration.
- Hamiltonian `H(q, p) = U(q) + K(p)`, `U = −log posterior`.
- **Leapfrog integrator** + Metropolis-Hastings accept step
  preserves detailed balance.

**NUTS** (No-U-Turn Sampler) auto-tunes trajectory length by
recursive doubling until the path U-turns; Stan's default sampler.

## When to use

- **Continuous, differentiable posteriors** with many parameters
  (SEM, hierarchical models, GPs).
- **Poor mixing** with Metropolis-Hastings or Gibbs.

## When NOT to use

- **Discrete parameters** — HMC needs gradients; use
  Metropolis-within-Gibbs.
- **Non-differentiable target** (indicator functions, hard
  constraints) — use reparameterisation or reject-augmented HMC.

## Files

- `python/hmc_nuts.py` — from-scratch HMC with leapfrog, fixed
  `(ε, L)`. Demo (2-D correlated Gaussian, ρ=0.8): **acceptance
  0.99**, recovered mean **(1.00, −0.49)** and Σ within 0.03 of
  truth after 2000 iterations, 500 burn-in.
- `r/hmc_nuts.R` — `rstan` / `cmdstanr` / `brms`, `rethinking`
  (R); `numpyro`, `pymc`, `blackjax`,
  `tensorflow-probability` (Python).

## Assumptions & caveats

- **Step size ε** — too large → divergences; too small → wasted
  compute. NUTS auto-tunes.
- **Mass matrix M** — set to posterior inverse-covariance for
  best efficiency (Stan does this by default during warm-up).
- **Warm-up / burn-in** — discard the first ~30-50 % of samples.
- **Convergence diagnostics** — R̂ across chains, effective
  sample size, divergent transitions.

## Related in this repo

- `mcmc-metropolis-hastings`, `gibbs-sampler` — non-gradient MCMC
  cousins.
- `bayesian-glms`, `bayesian-neural-network` — downstream users.
- `information-geometry` — HMC + natural gradient overlap.

## Run

```
python techniques/hmc-nuts/python/hmc_nuts.py
Rscript techniques/hmc-nuts/r/hmc_nuts.R
```

**Refs:** Duane, S., Kennedy, A.D., Pendleton, B.J., & Roweth, D. "Hybrid Monte Carlo." *Physics Letters B*, 1987; Neal, R.M. "MCMC using Hamiltonian dynamics." In *Handbook of Markov Chain Monte Carlo*, 2011; Hoffman, M.D. & Gelman, A. "The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." *JMLR*, 2014.

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

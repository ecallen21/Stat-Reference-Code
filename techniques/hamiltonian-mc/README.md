# Hamiltonian Monte Carlo (Reference §14.8)

MCMC that uses **gradient information** to make big, informed proposals across the posterior. Vastly outperforms random-walk MH in moderate/high dimension and on strongly correlated targets.

## Setup

Augment `θ` with a fictitious momentum `p ~ N(0, M)`. Define the Hamiltonian

```
H(θ, p) = U(θ) + K(p)         U = −log π(θ),    K = ½ pᵀ M⁻¹ p
```

Move `(θ, p)` along Hamilton's equations for `L` **leapfrog** steps of size `ε`; accept the endpoint with probability `min{1, exp(−ΔH)}`.

## Leapfrog integrator

```
p_{1/2} = p − (ε/2) ∇U(θ)
θ_1     = θ + ε M⁻¹ p_{1/2}
p_1     = p_{1/2} − (ε/2) ∇U(θ_1)
```

Volume-preserving and reversible — the two properties that keep the MH acceptance rule valid.

## Tuning

- Step size `ε` — target acceptance ≈ 0.65–0.9. Too big → poor acceptance; too small → wasted computation.
- Trajectory length `L` — long enough to explore, short enough not to U-turn.
- **NUTS** (No-U-Turn Sampler, Hoffman & Gelman 2014) picks `L` automatically by building a binary tree of leapfrog steps in both directions and stopping at the first U-turn; **dual averaging** tunes `ε` during warm-up. Production implementations: Stan, PyMC, NumPyro, Turing.jl.

## Files

- `python/hamiltonian_mc.py` — vanilla HMC with fixed `(ε, L)` and unit mass matrix; supports numerical or analytical gradients. Demo on a 3-D correlated Gaussian recovers means to ~0.01 and the full covariance to ~0.05.
- `r/hamiltonian_mc.R` — same leapfrog HMC in base R. For production HMC/NUTS use `rstan` or `cmdstanr`.

## When to use

- Continuous, moderate-to-high-dimensional posteriors with differentiable log-density.
- Strong parameter correlations that stall random-walk MH or Gibbs.
- Whenever you can get gradients (analytical or autodiff — JAX, PyTorch, Stan's autodiff, Zygote.jl).

## When to fall back

- Discrete parameters — HMC has no gradient there. Marginalize the discrete parameter out (Rao-Blackwellize) and sample the continuous marginal with HMC.
- Non-differentiable log-density (step functions, hard truncations at posterior modes).
- Very high curvature / funnel geometries — consider the non-centered parameterization first, then Riemannian HMC.

## Diagnostics

- Divergent transitions (energy jumps that indicate the leapfrog integrator can't follow the curvature).
- E-BFMI (Bayesian fraction of missing information).
- R-hat and ESS as with any MCMC.

## Run

```
python techniques/hamiltonian-mc/python/hamiltonian_mc.py
Rscript techniques/hamiltonian-mc/r/hamiltonian_mc.R
```

**Refs:** Duane, S., Kennedy, A.D., Pendleton, B.J. & Roweth, D. "Hybrid Monte Carlo." *Phys. Lett. B* 195(2), 216–222, 1987; Neal, R.M. "MCMC using Hamiltonian dynamics." In *Handbook of Markov Chain Monte Carlo*, CRC, 2011; Hoffman, M.D. & Gelman, A. "The No-U-Turn Sampler." *JMLR* 15(1), 1593–1623, 2014.

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

# Bouncy Particle Sampler (Reference §47.359)

Bouchard-Cote, Vollmer & Doucet (2018 JASA). Piecewise-
deterministic Markov process (PDMP) MCMC. A particle moves
in straight lines at unit speed; direction reflects off the
gradient of `−log π` at random times drawn from a Poisson
process with rate:

```
λ(x, v) = max(0, ⟨v, ∇U(x)⟩),   U(x) = −log π(x)
```

An extra refresh rate `λ_ref` re-randomises the velocity to
keep ergodicity. Non-reversible → better asymptotic variance
than reversible MCMC in many settings.

## Files

- `python/bouncy_particle_sampler.py` — correlated 3-D
  Gaussian target. 5 000 bounce events give sample mean
  ≈ (0.018, 0.013, −0.007) and sample covariance to
  Frobenius rel-error 0.12 of the true `Σ` after burn-in of
  500 events.
- `r/bouncy_particle_sampler.R` — `RZigZag::BouncyParticle`,
  `PDMPFlux` (R); `pdmp_jax`, from-scratch (Python).

## When to use

- **Big-data Bayesian inference** — the local implementation
  visits subsets of the data at each bounce (super-efficient
  scaling with n).
- **Log-concave posteriors** — analytic upper bounds on λ let
  Poisson thinning skip data.
- **High-dimensional smooth targets** — non-reversibility
  gives asymptotic-variance gains over MALA/HMC in some
  settings.

## When NOT to use

- **Discrete or heavy-tailed targets** — PDMPs are designed
  for continuous smooth densities.
- **When a simple HMC / NUTS works** — those are easier to
  implement and tune.
- **When gradient is unavailable** — the bounce rule needs it.

## Assumptions & caveats

- **Poisson thinning bound `upper_M`** — needs an upper
  bound on `⟨v, ∇U⟩`; too loose ⇒ many rejections, too
  tight ⇒ invalidates sampling.
- **Refresh rate λ_ref** — non-zero refresh needed for
  ergodicity in dim ≥ 2; typical choice matches the target's
  correlation time.
- **Bounce reflection** — `v ← v − 2 (v·∇U / ‖∇U‖²) ∇U`.
- **Zig-Zag sampler** — a related PDMP that bounces
  coordinate-wise; simpler `λ` bounds.
- **Continuous-time sampling** — samples are trajectory
  segments, not discrete points; grid-averaging or
  linear-interp along events gives estimators.

## Related in this repo

- `hmc-nuts`, `hamiltonian-mc`, `mala-langevin`,
  `stochastic-gradient-mcmc` — gradient-informed reversible
  MCMC alternatives.
- `parallel-tempering-mcmc`, `elliptical-slice-sampling` —
  advanced MCMC neighbours.
- `stein-variational-gradient`, `variational-inference`
  — deterministic alternatives.
- `sequential-monte-carlo` (`particle-filter-smc`) —
  particle-flow siblings.

## Run

```
python techniques/bouncy-particle-sampler/python/bouncy_particle_sampler.py
Rscript techniques/bouncy-particle-sampler/r/bouncy_particle_sampler.R
```

**Refs:** Bouchard-Cote, A., Vollmer, S.J. and Doucet, A. "The bouncy particle sampler: a non-reversible rejection-free Markov chain Monte Carlo method." *J. Amer. Statist. Assoc.*, 113(522): 855-867, 2018.

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

# Zig-Zag Sampler (Reference §47.371)

Bierkens, Fearnhead & Roberts (2019). Piecewise-deterministic
Markov process (PDMP) MCMC where each coordinate has an
independent velocity `v_i ∈ {−1, +1}`. Coordinate `i`'s
velocity flips with rate

```
λ_i(x, v) = max(0, v_i · ∂U/∂x_i(x))     (U = −log π)
```

Times to flip drawn by Poisson thinning against an upper
bound `M_i`. Simpler than the bouncy particle sampler
(coordinate-wise bounces) and often SUPERIOR on very high-
dimensional targets because sub-sampling of the gradient is
possible with exact stationary distribution.

## Files

- `python/zig_zag_sampler.py` — 5-D independent standard
  Normal. 5 000 zig-zag events give sample means within
  ±0.04 of 0 and sample variances 1.14-1.27 (target 1);
  event-state statistics slightly overshoot because the
  process spends more time near turning points.
- `r/zig_zag_sampler.R` — `RZigZag::ZigZagLogistic`,
  `RZigZag::ZigZagGaussian`, `PDMPFlux` (R); `pdmp_jax`,
  from-scratch (Python).

## When to use

- **High-dimensional Bayesian targets** — evidence of
  favourable scaling with dimension vs reversible MCMC.
- **Data sub-sampling** — Zig-Zag with sub-sampling still
  targets the exact posterior (unlike SGLD).
- **Log-concave targets** — analytic upper bounds on
  `∂U/∂x_i` make Poisson thinning cheap.

## When NOT to use

- **Discrete or heavy-tailed targets** — PDMPs designed for
  smooth continuous densities.
- **When code simplicity trumps efficiency** — plain HMC /
  NUTS is battle-tested.
- **Multi-modal targets** — like other MCMC, gets stuck
  without tempering.

## Assumptions & caveats

- **Upper bounds on per-coordinate rate** — need `M_i ≥
  sup_{x, v_i} v_i · ∂U/∂x_i(x)`; too loose ⇒ many rejections.
- **Sub-sampling with control variate** — the `Zig-Zag with
  Data-Sub-sampling` (ZZ-SS) variant is exact under mild
  assumptions.
- **Continuous-time output** — samples along piecewise-linear
  trajectory; discrete event-time states approximate the
  integrated statistic but can bias variance estimates.
- **Ergodicity** — depends on `λ_i > 0` almost everywhere;
  add a refresh rate on log-concave targets.

## Related in this repo

- `bouncy-particle-sampler`, `mala-langevin`, `hmc-nuts`,
  `hamiltonian-mc` — related gradient MCMC.
- `stochastic-gradient-mcmc`, `parallel-tempering-mcmc` —
  scalable-MCMC neighbours.
- `stein-variational-gradient` — deterministic particle
  alternative.
- `verlet-leapfrog-symplectic` — HMC's integrator kernel.

## Run

```
python techniques/zig-zag-sampler/python/zig_zag_sampler.py
Rscript techniques/zig-zag-sampler/r/zig_zag_sampler.R
```

**Refs:** Bierkens, J., Fearnhead, P. and Roberts, G. "The zig-zag process and super-efficient sampling for Bayesian analysis of big data." *Ann. Statist.*, 47(3): 1288-1320, 2019.

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

# Parallel Tempering (Reference §47.358)

Geyer (1991); refined by Earl & Deem (2005). Run K parallel
MCMC chains at temperatures `1 = T₁ < T₂ < … < T_K`, each
targeting `π^(1/T_k)`. Hotter chains explore more freely; the
cold chain is the one of interest. Every few steps propose
swapping states of ADJACENT chains with Metropolis-Hastings
probability:

```
α = min(1, (π(x_hot) / π(x_cold))^(1/T_cold − 1/T_hot))
```

Cures the multi-modal mixing problem that traps plain
Metropolis-Hastings — an ergodic-in-practice sampler when a
single chain would need exponentially long to hop between
modes.

## Files

- `python/parallel_tempering_mcmc.py` — bimodal target
  `½ N(−3, 1) + ½ N(+3, 1)`. Single-chain MH is stuck in the
  left mode with fraction 0.356 (target 0.5); parallel
  tempering with 5 chains at `T ∈ {1, 2, 4, 8, 16}` recovers
  fraction 0.494 and cold-chain mean 0.015 (target 0.0).
  Swap rates ~ 0.8 across adjacent temperatures.
- `r/parallel_tempering_mcmc.R` — `nimble` (tempered MCMC in
  NIMBLE), custom loop (R); `emcee.PTSampler` (deprecated
  reference), `blackjax.tempered_smc`, from-scratch (Python).

## When to use

- **Multi-modal posteriors** — mixture models, phylogenetics,
  protein folding, Bayesian mixture-of-experts.
- **Rough energy landscapes** — magnetic systems, glasses,
  RNA / molecular simulation.
- **Model-selection posteriors** — trans-dimensional and
  RJ-MCMC benefit from tempering.

## When NOT to use

- **Unimodal targets** — plain HMC / NUTS is faster.
- **When you can afford NUTS + reparameterisation** — a good
  parameterisation often removes the multi-modality.
- **Very tight computational budgets** — running K chains
  multiplies cost.

## Assumptions & caveats

- **Temperature ladder** — geometric spacing is a standard
  starting choice; tune for swap rates 20-40 %.
- **Adaptive ladders** — Katzgraber's algorithm and Vousden
  (2016) tune spacing on-the-fly.
- **Global convergence diagnostics** — R-hat still applies,
  but only on the cold chain.
- **Only cold-chain samples represent π** — hotter chains
  are auxiliary.
- **Communication cost** — parallelise the CHAINS across
  cores; swap step needs sync.

## Related in this repo

- `mcmc-metropolis-hastings`, `hmc-nuts`, `hamiltonian-mc`,
  `adaptive-metropolis-haario`, `mala-langevin` — base
  samplers each chain can use.
- `bouncy-particle-sampler`, `stein-variational-gradient`,
  `sequential-monte-carlo`-style methods — alternative
  approaches to multi-modality.
- `bayesian-model-comparison`, `bridge-sampling-evidence` —
  post-hoc uses of tempered chains for marginal likelihood.
- `simulated-annealing` — deterministic cousin.

## Run

```
python techniques/parallel-tempering-mcmc/python/parallel_tempering_mcmc.py
Rscript techniques/parallel-tempering-mcmc/r/parallel_tempering_mcmc.R
```

**Refs:** Geyer, C.J. "Markov chain Monte Carlo maximum likelihood." *Comput. Sci. Statist.*, 23: 156-163, 1991; Earl, D.J. and Deem, M.W. "Parallel tempering: theory, applications, and new perspectives." *Phys. Chem. Chem. Phys.*, 7: 3910-3916, 2005.

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

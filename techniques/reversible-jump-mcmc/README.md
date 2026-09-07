# Reversible-Jump MCMC (Reference §25.6)

Green (1995). Extends Metropolis-Hastings to a space where the
**dimension** of the parameter vector can change, allowing joint
sampling over models AND their parameters:

    x = (k, θ_k),   θ_k ∈ ℝ^{n_k}

## Move types

- **Within-model MH** in model `k`.
- **Trans-dimensional jump** `k → k'` using a dimension-matching
  bijection `θ_{k'} = g(θ_k, u)` with `u ~ q(u)` and Jacobian `J`.

Acceptance ratio (jump):

    A = [π(k', θ_{k'}) / π(k, θ_k)] · [q(u') / q(u)] · |J|

## Files

- `python/reversible_jump_mcmc.py` — RJ-MCMC for the number of
  Gaussian mixture components `k ∈ {1, 2, 3}` from scratch with
  birth / death moves. Demo (n=200 from 2-component mixture at
  μ = ±2): posterior `P(k = 2) = 0.997`, correctly identifying the
  true model.
- `r/reversible_jump_mcmc.R` — `rjmcmc`, `coda`, `MCMCglmm` (R);
  from-scratch numpy / pymc custom sampler (Python).

## When to use

- **Bayesian model choice** — variable selection, mixture component
  number, spline knot number, change-point number.
- **Simultaneous parameter + model inference** — the whole posterior
  including model uncertainty.
- **Complex model spaces** without tractable marginal likelihoods —
  RJ mixes model and parameter posteriors without needing an
  explicit `p(y | k)`.

## When NOT to use

- **Few candidate models with tractable marginal likelihoods** —
  compute Bayes factors directly.
- **High-dimensional θ** — the dimension-matching proposal is hard
  to design; consider Gibbs variable selection or spike-and-slab.
- **Simple model comparisons** — bridge sampling or thermodynamic
  integration may be simpler.

## Assumptions & caveats

- **Detailed balance** — the acceptance ratio must include the
  Jacobian of the dimension-matching bijection; mistakes here bias
  the posterior.
- **Proposal design** dominates efficiency — birth / death,
  split / merge, and "reversible pair" proposals each have their
  place.
- **Convergence diagnostics** — the chain visits varying-dimension
  spaces; use trans-D diagnostics (visit frequencies, split-R̂ on
  scalar summaries).
- **Label switching** — when mixture components are exchangeable,
  post-process traces (relabel or use invariant summaries).

## Related in this repo

- `mcmc-metropolis-hastings`, `gibbs-sampler`, `hmc-nuts` — the
  fixed-dimension samplers.
- `bayesian-model-averaging`, `bayesian-model-comparison`,
  `bridge-sampling-evidence` — alternative model-choice tools.
- `dirichlet-process-mixture` — a related "unknown k" sampler that
  avoids RJ.

## Run

```
python techniques/reversible-jump-mcmc/python/reversible_jump_mcmc.py
Rscript techniques/reversible-jump-mcmc/r/reversible_jump_mcmc.R
```

**Refs:** Green, P.J. "Reversible jump Markov chain Monte Carlo computation and Bayesian model determination." *Biometrika*, 82(4): 711-732, 1995; Richardson, S. & Green, P.J. "On Bayesian analysis of mixtures with an unknown number of components." *JRSS-B*, 59(4): 731-792, 1997.

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

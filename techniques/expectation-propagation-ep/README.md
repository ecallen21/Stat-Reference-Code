# Expectation Propagation (EP) (Reference §47.120)

Minka (2001). Approximate a posterior `p(θ) ∝ prior · ∏ᵢ fᵢ(θ)` by
a member `q(θ)` of an exponential family. Iterate over sites:

    q_{−i}(θ) ∝ q(θ) / f̃ᵢ(θ)                       (cavity)
    q^{new}(θ) = KL-project(q_{−i}(θ) fᵢ(θ))      (moment match)
    f̃ᵢ^{new}(θ) ∝ q^{new}(θ) / q_{−i}(θ)          (site update).

Foundation of Gaussian-process classification, Bayesian linear
regression under non-conjugate likelihoods, and Laplace-EP hybrids.

## Files

- `python/expectation_propagation_ep.py` — from-scratch EP for
  Bayesian PROBIT regression with 1-D prior N(0, σ²). Demo (n=300,
  β_true = 1.5):
  - EP posterior mean 1.334, sd 0.087
  - Ground-truth grid mean 1.339, sd 0.136 — mean matches, EP
    slightly understates sd.
- `r/expectation_propagation_ep.R` — `EPGLM`, `GPclassify`
  (limited R); `GPy`, `pyprobml-utils`, from-scratch (Python).

## When to use

- **Non-conjugate posteriors** with tractable per-site tilted
  moments — probit, ordinal, robust regression.
- **Gaussian-process classification** — EP is the standard
  approximate-inference algorithm.
- **Sparse Bayesian learning / RVM** — pointwise EP for site-relaxation.
- **When VI is available but slower / less accurate**.

## When NOT to use

- **Highly multimodal posteriors** — EP moment matches, may
  concentrate on one mode.
- **Very heavy-tailed likelihoods** — moment matching fails to
  converge.
- **When exact MCMC is affordable** and gold-standard needed.

## Assumptions & caveats

- **Convergence** not guaranteed — damping / power-EP often
  helps.
- **Moment matching** requires tractable tilted moments; symbolic
  for standard likelihoods, numerical otherwise.
- **Site parameterisation** — mean & precision (or natural
  params) matter for numerical stability.
- **Alternatives**: VI (KL(q‖p)), Laplace, HMC — trade speed for
  accuracy.

## Related in this repo

- `variational-inference`, `laplace-approximation`,
  `hmc-nuts`, `hamiltonian-mc`, `stochastic-gradient-mcmc` —
  approximate-inference toolkit.
- `gaussian-process-regression`,
  `gaussian-process-latent-variable-model`,
  `sparse-gaussian-process`, `bayesian-linear-regression`,
  `bayesian-neural-network`, `bayesian-glms`,
  `bayesian-hierarchical-models` — Bayesian model families.
- `mcmc-metropolis-hastings`, `slice-sampler`, `gibbs-sampler`,
  `reversible-jump-mcmc` — MCMC alternatives.

## Run

```
python techniques/expectation-propagation-ep/python/expectation_propagation_ep.py
Rscript techniques/expectation-propagation-ep/r/expectation_propagation_ep.R
```

**Refs:** Minka, T.P. "Expectation Propagation for approximate Bayesian inference." *UAI*, 2001; Rasmussen, C.E. & Williams, C.K.I. *Gaussian Processes for Machine Learning.* MIT Press, 2006, chapter 3.6.

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

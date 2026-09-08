# Elliptical Slice Sampling (Reference §25.12)

Murray, Adams & MacKay (2010, AISTATS). MCMC tailored to
**Gaussian prior + arbitrary likelihood**:

    prior:      f ~ N(0, K)
    posterior:  p(f | data) ∝ L(f) · N(f; 0, K)

## Update

1. Draw auxiliary `ν ~ N(0, K)`.
2. Slice-height `log u = log L(f) + log(Uniform)`.
3. Angle `θ ~ U(0, 2π)`; bracket `[θ − 2π, θ]`.
4. Propose `f' = f cos(θ) + ν sin(θ)`.
5. Accept if `log L(f') > log u`; else shrink bracket to exclude
   `θ` and retry.

**No tuning** (unlike MH); preserves Gaussian prior exactly.

## Files

- `python/elliptical_slice_sampling.py` — ESS from scratch on a
  toy GP-regression posterior (RBF kernel, `n=15`). Demo: ESS
  posterior mean vs analytic GP mean correlate at 0.9994; MSE vs
  the true sine 0.022.
- `r/elliptical_slice_sampling.R` — `ess`, `spBayes`, GP HMC
  alternatives (R); `gpytorch`, `pymc`, from-scratch (Python).

## When to use

- **GP-regression posterior** — non-Gaussian likelihood (probit,
  Poisson, negative binomial) with GP prior.
- **Factor / spatial random-field models** with a Gaussian latent.
- **Bayesian neural networks** with Gaussian weight priors + non-
  Gaussian likelihood.
- **Whenever prior is Gaussian** and likelihood is cheap.

## When NOT to use

- **Non-Gaussian prior** — ESS is specifically Gaussian-prior; use
  HMC / MH.
- **Very expensive likelihood** — ESS may need many likelihood
  evaluations per accept if bracket shrinks slowly.
- **Very correlated components** — extend to parallel / multiple-
  update ESS (Nishihara et al. 2014).

## Assumptions & caveats

- **Ergodic despite shrinkage** — bracket-shrink strategy is
  provably valid.
- **Cholesky of `K`** required — for large `n`, use sparse /
  Kronecker decomposition.
- **Diminishing bracket** guarantees termination; typical ESS
  needs 2-5 likelihood evaluations per step.
- **No step size** to tune, but the underlying prior scale matters
  — mis-specified `K` yields poor mixing.

## Related in this repo

- `slice-sampler`, `mcmc-metropolis-hastings`, `hmc-nuts`,
  `adaptive-metropolis-haario` — MCMC family.
- `gaussian-process-regression`,
  `gaussian-process-latent-variable-model` — GP consumers.
- `variational-inference`, `nested-sampling` — Bayesian alternatives.

## Run

```
python techniques/elliptical-slice-sampling/python/elliptical_slice_sampling.py
Rscript techniques/elliptical-slice-sampling/r/elliptical_slice_sampling.R
```

**Refs:** Murray, I., Adams, R.P. & MacKay, D.J.C. "Elliptical slice sampling." *AISTATS*, 2010; Nishihara, R. et al. "Parallel MCMC with generalized elliptical slice sampling." *JMLR*, 15: 2087-2112, 2014.

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

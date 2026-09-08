# Rejection Sampling (Reference §47.45)

von Neumann (1951). To sample from density f, choose a proposal g
and constant M such that f(x) ≤ M · g(x) for all x. Iterate:

    1. Draw x ~ g.
    2. Accept with probability f(x) / (M g(x)); else reject.

Accepted x has density f. Expected acceptance rate = 1/M (normalised
f). Extensions: adaptive rejection (Gilks-Wild 1992) for log-concave
densities builds the envelope online.

## Files

- `python/rejection_sampling.py` — generic rejection sampler +
  demo drawing from a truncated N(0, 1) on [0, 3] via
  Exp(1) proposal (M = e^{½} ≈ 1.65). Achieves 0.40 acceptance
  rate; KS-stat vs analytical truncated normal = 0.0045 on n=20 000
  samples.
- `r/rejection_sampling.R` — `ars`, `Runuran::pinv.new` (universal
  inversion) in R; `scipy.stats.rvs`, from-scratch (Python).

## When to use

- **Univariate f known up to constant** — Bayesian conjugate
  updates or intractable normalisers.
- **Bounded log-density ratio** — need finite M.
- **When you don't need HMC / MCMC** — rejection gives IID samples.

## When NOT to use

- **High-dimensional f** — M grows exponentially in d (curse of
  dimensionality); use MCMC.
- **Heavy-tailed target with light-tailed proposal** — M is infinite;
  swap proposals.
- **Very sharp f** — acceptance rate crashes; use importance
  sampling with SNIS, or MH.

## Assumptions & caveats

- **Envelope constant M** — must be finite; wrong M → biased draws
  (M too small drops mass; M too large just wastes proposals).
- **Adaptive rejection** requires log-concavity; otherwise use
  squeeze-based ARS-metropolis.
- **Truncated targets** — the acceptance ratio should include the
  indicator on the support.
- **Efficiency ∝ 1/M** — for common densities M can be tightened
  analytically.

## Related in this repo

- `mcmc-metropolis-hastings`, `hmc-nuts`, `slice-sampler` — general
  posterior sampling.
- `importance-sampling`, `quasi-monte-carlo-sobol`,
  `antithetic-control-variates` — variance-reduced Monte Carlo.
- `abc-approximate-bayesian` — likelihood-free rejection sampler.
- `latin-hypercube-sampling` — stratified alternative.

## Run

```
python techniques/rejection-sampling/python/rejection_sampling.py
Rscript techniques/rejection-sampling/r/rejection_sampling.R
```

**Refs:** von Neumann, J. "Various techniques used in connection with random digits." *NBS Applied Math Series* 12: 36-38, 1951; Gilks, W.R. & Wild, P. "Adaptive rejection sampling for Gibbs sampling." *Appl Stat* 41(2): 337-348, 1992.

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

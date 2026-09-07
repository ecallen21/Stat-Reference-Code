# Importance Sampling (Reference §45.6)

Geweke (1989), Vehtari-Gelman-Gabry (2017). Estimate `E_p[f(X)]`
using draws from a different **proposal** `q`:

```
E_p[f(X)] = E_q[ f(X) · p(X)/q(X) ]
```

with **self-normalised** form
`Σ f w / Σ w`, `w = p/q`, when p is known only up to a normalising
constant.

**Effective sample size** (Kong 1992):
`ESS = (Σ w)² / Σ w²` — diagnoses proposal quality; `ESS << n`
means the proposal is a poor fit to the target.

## When to use

- **Reweight posterior draws** to a different prior / model
  without re-sampling.
- **Cross-validation from a single MCMC fit** — PSIS-LOO
  (Vehtari-Gelman 2017).
- **Rare-event probability** with a tailored proposal.

## When NOT to use

- **Very high-dimensional targets** — importance ratios collapse
  to point masses; use SMC or annealed IS.
- **Heavy-tailed target with light-tailed proposal** — infinite
  variance; Pareto-smoothed IS (PSIS) helps.

## Files

- `python/importance_sampling.py` — SNIS + ESS (custom). Demo:
  target N(0,1), f(x)=x⁴, true E=3. `q=N(0,2)`: **n=100 000 →
  est 3.003, ESS ≈ 66 000**. Bad proposal `q=N(0,0.5)`:
  **est 1.34** (biased) with ESS ≈ 1 400.
- `r/importance_sampling.R` — `loo::psis`, `rstan`, custom (R);
  `arviz.psislw`, `numpyro`, `scipy` + custom (Python).

## Assumptions & caveats

- **Proposal support ⊇ target support** — zero-weight anywhere
  the target has mass biases the estimator.
- **Log-domain arithmetic** — subtract max before exponentiating
  weights.
- **PSIS diagnostics** — a Pareto k > 0.7 flags unreliable
  reweighting; refit or resample.
- **Sequential IS / SMC** for high-dim or evolving targets.

## Related in this repo

- `mcmc-metropolis-hastings`, `hmc-nuts`, `gibbs-sampler` — MCMC
  alternatives.
- `abc-approximate-bayesian` — simulation-based inference
  cousin.

## Run

```
python techniques/importance-sampling/python/importance_sampling.py
Rscript techniques/importance-sampling/r/importance_sampling.R
```

**Refs:** Geweke, J. "Bayesian inference in econometric models using Monte Carlo integration." *Econometrica*, 1989; Vehtari, A., Gelman, A., & Gabry, J. "Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC." *Statistics and Computing*, 2017.

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

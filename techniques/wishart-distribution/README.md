# Wishart Distribution (Reference §47.395)

Wishart (1928). Distribution of the scatter matrix
`S = XᵀX` for `X ~ N_n(0, Σ)`, `n ≥ p`. Written
`S ~ W_p(n, Σ)`.

- `E[S] = n · Σ`
- Conjugate prior for the PRECISION matrix in a multivariate
  Normal model.
- Inverse-Wishart is the corresponding prior for the
  COVARIANCE.

Bartlett decomposition gives an efficient sampler:
`S = LAAᵀLᵀ` with `L = chol(Σ)`, `A_ii = χ_{n−i+1}`,
lower-triangle `A_ij ~ N(0, 1)`.

## Files

- `python/wishart_distribution.py` — 3×3 target `Σ` at
  `n = 20`, 5 000 draws via Bartlett. Empirical `E[S]`
  matches `n·Σ` to Frobenius rel-error 0.005.
- `r/wishart_distribution.R` — `stats::rWishart`,
  `MCMCpack::rwish`, `LaplacesDemon` (R);
  `scipy.stats.wishart`, from-scratch Bartlett (Python).

## Where else it appears in this repo

- `bayesian-hierarchical-models`, `bayesian-glms` — Wishart
  / inverse-Wishart as covariance prior.
- `covariance-estimation-highdim` — Wishart is the sampling
  distribution of the sample covariance.
- `gaussian-graphical-model` — precision-matrix prior;
  Cholesky and G-Wishart variants.
- `bayesian-linear-regression`, `probabilistic-pca` — MVN
  conjugate updates use Wishart.
- `mcmc-metropolis-hastings`, `hmc-nuts` — samplers that
  can leverage the conjugacy.

## Assumptions & caveats

- **Degrees of freedom** — must satisfy `n ≥ p` for a
  non-degenerate Wishart.
- **Inverse-Wishart** — S⁻¹ ~ IW_p(n, Σ⁻¹); common in
  Bayesian workflows.
- **Off-diagonal correlations** — Wishart induces
  correlations in the sample covariance; ignore at your
  peril when doing simulation-based coverage checks.
- **Numerical caveats** — samples for large p can be
  ill-conditioned; use log-Cholesky parameterisation for
  posterior sampling.
- **G-Wishart** — restricted-support Wishart on a graph
  structure; conjugate to Gaussian graphical models.

## Run

```
python techniques/wishart-distribution/python/wishart_distribution.py
Rscript techniques/wishart-distribution/r/wishart_distribution.R
```

**Refs:** Wishart, J. "The generalized product moment distribution in samples from a normal multivariate population." *Biometrika*, 20A: 32-52, 1928; Muirhead, R.J. *Aspects of Multivariate Statistical Theory*, Wiley, 1982.

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

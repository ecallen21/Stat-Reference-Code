# Multivariate Normal Distribution (Reference §47.396)

Foundational distribution:

```
X ~ N_p(μ, Σ)
f(x) = (2π)^(−p/2) |Σ|^(−1/2) exp(−½ (x − μ)ᵀ Σ⁻¹ (x − μ))
```

Key properties:

- Marginals, conditionals, and linear combinations are all
  multivariate Normal.
- Quadratic form `(x − μ)ᵀ Σ⁻¹ (x − μ) ~ χ²_p`.
- Sample via Cholesky: `X = μ + LZ`, `Z ~ N(0, I)`,
  `LLᵀ = Σ`.

## Files

- `python/multivariate_normal_distribution.py` — Cholesky
  sampler, MLE `(μ̂, Σ̂)` from 10 000 draws in p=3. Recovers
  `μ` to 2 decimals and every entry of `Σ` to 2-3 decimals.
  Verifies the χ²_3 quadratic-form identity (mean 3.005 vs
  theory 3.00; variance 5.9 vs theory 6.0).
- `r/multivariate_normal_distribution.R` — `MASS::mvrnorm`,
  `mvtnorm::rmvnorm` / `dmvnorm` (R);
  `scipy.stats.multivariate_normal`, from-scratch (Python).

## Where else it appears in this repo

- `gaussian-process-regression`, `sparse-gaussian-process` —
  GP is an MVN over function values.
- `state-space-kalman`, `extended-kalman-filter`,
  `unscented-kalman-filter`, `ensemble-kalman-filter` —
  MVN posterior in the recursive filter.
- `bayesian-linear-regression`, `bayesian-hierarchical-models`,
  `probabilistic-pca`, `bayesian-neural-network` — MVN
  priors on parameters.
- `canonical-correlation`, `manova`, `hotellings-t2`,
  `mahalanobis-distance-matching`, `lda-qda` — multivariate
  Gaussian inference.
- `mcmc-metropolis-hastings`, `mala-langevin`, `hmc-nuts` —
  MVN proposals and targets.

## Assumptions & caveats

- **PSD covariance** — `Σ ⪰ 0`; if singular, use pseudo-inverse
  or reduce to lower-dim marginal.
- **Cholesky vs eigendecomposition** — Cholesky is O(p³/3);
  eigendecomp gives numerically stable square-roots at 2×
  cost.
- **Conditional distribution** —
  `X_A | X_B = x_B ~ N(μ_A + Σ_AB Σ_BB⁻¹ (x_B − μ_B),
                       Σ_AA − Σ_AB Σ_BB⁻¹ Σ_BA)`.
- **Fisher information** — Cramér-Rao lower bound is
  attained by the MLE at both mean and covariance.
- **Robustness** — MVN is NOT robust to outliers; use
  Student-t or MCD for heavy-tail-safe inference.

## Run

```
python techniques/multivariate-normal-distribution/python/multivariate_normal_distribution.py
Rscript techniques/multivariate-normal-distribution/r/multivariate_normal_distribution.R
```

**Refs:** Muirhead, R.J. *Aspects of Multivariate Statistical Theory*, Wiley, 1982; Anderson, T.W. *An Introduction to Multivariate Statistical Analysis*, 3rd ed., Wiley, 2003.

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

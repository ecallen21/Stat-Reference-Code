# Gibbs Sampler (Reference §14.7)

Special case of Metropolis-Hastings where each proposal is drawn from the exact **full conditional**, giving acceptance probability 1. When every full conditional is tractable (conjugate structure), Gibbs sweeps are cheap and mix well without tuning.

## Normal – Inverse-Gamma prototype

```
y_i | μ, σ²  ~ Normal(μ, σ²)
μ | σ²       ~ Normal(μ₀, σ²/κ₀)
σ²           ~ InvGamma(α₀, β₀)
```

Full conditionals:

```
μ | σ², y   ~ Normal(μ_n, σ²/κ_n),      κ_n = κ₀ + n,  μ_n = (κ₀μ₀ + nȳ)/κ_n
σ² | μ, y   ~ InvGamma(α₀ + n/2,  β₀ + ½ Σ(yᵢ−μ)²)
```

## Two-level hierarchical Normal (8-schools flavor)

```
y_j    ~ Normal(θ_j, σ_j²)          σ_j known
θ_j    ~ Normal(μ, τ²)              (shrinks θ_j toward μ)
μ      ~ Normal(0, 100²)
τ²     ~ InvGamma(0.5, 0.5)
```

Each parameter's full conditional stays Normal or InvGamma → straight Gibbs.

## Files

- `python/gibbs_sampler.py` — Normal-InvGamma Gibbs (recovers posterior mean of μ = 3.83 vs sample mean 3.83) and hierarchical 8-schools Gibbs (extreme values 28 and −3 shrunk to ≈ 7.4 toward the overall mean).
- `r/gibbs_sampler.R` — base-R Normal-InvGamma Gibbs.

## When to use

- Conjugate hierarchical models where every full conditional is standard.
- Latent-variable models (mixture models, probit regression, missing-data augmentation).
- As a subroutine inside MH-within-Gibbs when only some full conditionals are tractable.

## When to prefer HMC

- Strongly correlated posteriors — Gibbs mixes badly along the correlation direction.
- Non-conjugate hierarchies where each full-conditional draw itself needs MH.

## Diagnostics

- Discard burn-in (default here: first 20% of iterations).
- Check trace plots of `μ`, `σ²`, and each `θ_j`.
- Effective sample size and Gelman-Rubin R-hat over multiple chains.

## Run

```
python techniques/gibbs-sampler/python/gibbs_sampler.py
Rscript techniques/gibbs-sampler/r/gibbs_sampler.R
```

**Refs:** Geman, S. & Geman, D. "Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images." *IEEE Trans. Pattern Anal. Mach. Intell.* PAMI-6(6), 721–741, 1984; Gelfand, A.E. & Smith, A.F.M. "Sampling-based approaches to calculating marginal densities." *JASA* 85(410), 398–409, 1990; Rubin, D.B. "Estimation in parallelized randomized experiments." *J. Educ. Stat.* 6(4), 377–401, 1981 (8-schools).

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

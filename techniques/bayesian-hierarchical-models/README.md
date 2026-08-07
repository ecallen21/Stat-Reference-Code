# Bayesian Hierarchical Models (Reference §14.15, §14.16)

Groups share statistical strength through a common prior over group-level parameters — classic **partial pooling**. The canonical example is Rubin's "8 schools":

```
y_j     ~ Normal(θ_j, σ_j²)             j = 1, ..., J   (σ_j known)
θ_j     ~ Normal(μ, τ²)                 (group-level prior)
μ       ~ Normal(0, 100²)               (diffuse hyperprior on mean)
τ       ~ HalfCauchy(5) or InvGamma(0.5, 0.5)   (weakly informative)
```

Compared to fully-pooled (`θ_j = μ`) or unpooled (`θ_j = y_j`) fits, the hierarchical posterior mean shrinks each `θ_j` toward `μ` by an amount that depends on `σ_j` and `τ`:

```
θ̂_j ≈ (y_j / σ_j² + μ / τ²) / (1/σ_j² + 1/τ²)
```

Imprecise groups shrink more; precise groups shrink less.

## Extensions

- **Random-intercepts / random-slopes regression** — `β_j ~ Normal(μ_β, Σ_β)`.
- **Meta-analysis** — study-level effect sizes with known SEs.
- **Multilevel logistic / Poisson** — swap the Normal likelihood; see `bayesian-glms`.

## Files

- `python/bayesian_hierarchical_models.py` — full Gibbs sampler on the hierarchical Normal model. On Rubin's 8-schools data, posterior means (7.0 – 8.1) all lie between the raw values (−3 to 28) and the fully-pooled mean (7.7). Also includes a side-by-side comparison of unpooled / pooled / hierarchical estimators.
- `r/bayesian_hierarchical_models.R` — same Gibbs sampler in base R. Production: `rstanarm::stan_glmer`, `brms::brm`.

## When to use

- Grouped data where each group has few observations but many groups share structure.
- Multi-site clinical trials, multi-lab experiments, multi-school educational studies.
- Small-area estimation.
- Meta-analysis of study effect sizes.

## Parameterization tip

For centered small-`τ` regimes the posterior develops the famous "funnel" shape. The **non-centered parameterization** helps HMC / NUTS:

```
θ_j = μ + τ · z_j        z_j ~ Normal(0, 1)
```

## Assumptions

- Group-level parameters `θ_j` are exchangeable in the prior — no reason `a priori` to distinguish groups.
- Chosen hyperprior on `τ` matters when the number of groups is small (say J ≤ 5); use a proper weakly informative HalfCauchy(scale) with a domain-scaled `scale`.

## Run

```
python techniques/bayesian-hierarchical-models/python/bayesian_hierarchical_models.py
Rscript techniques/bayesian-hierarchical-models/r/bayesian_hierarchical_models.R
```

**Refs:** Rubin, D.B. "Estimation in parallelized randomized experiments." *J. Educ. Stat.* 6(4), 377–401, 1981; Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., CRC, 2013 (Ch 5); Gelman, A. "Prior distributions for variance parameters in hierarchical models." *Bayesian Anal.* 1(3), 515–534, 2006.

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

# Dirichlet Process Gaussian Mixture (Reference §14.31)

Bayesian nonparametric mixture model where the number of components is **infinite a priori** but only a finite (data-driven) number is used. Removes the "how many clusters?" question from `k-means` / GMM by putting a prior over partitions.

## Setup

```
G     ~ DP(α, G_0)                    Dirichlet process prior
θ_i   ~ G
y_i   ~ F(θ_i)                        (Gaussian emission)
```

`α` = concentration parameter (higher → more clusters); `G_0` = base measure (typically Normal-Inverse-Gamma for Gaussian components).

## Chinese Restaurant Process (CRP)

Existing "customer" sits at table `k` with probability `n_k / (n − 1 + α)`, opens a new table with probability `α / (n − 1 + α)`. Priors on partitions follow this exchangeable rule.

## Neal's Algorithm 3 Gibbs sampler

For each observation `i`:

1. Remove `i` from its current cluster.
2. Compute posterior probability for each existing cluster `k`: `n_k · f(y_i | θ_k, θ obs)`.
3. Compute probability for a new cluster: `α · ∫ f(y_i | θ) G_0(dθ)`.
4. Sample the new cluster assignment; update sufficient statistics.

With conjugate `G_0`, `f` and `∫f dG_0` are closed-form (Student-t predictive under Normal-Inverse-Gamma).

## Files

- `python/dirichlet_process_mixture.py` — collapsed CRP Gibbs with Normal-Inverse-Gamma base measure. Demo on 3 well-separated Gaussians (n = 200): discovers 5 clusters (3 dominant + 2 tiny singletons); classifies all three true classes at ≥ 93% accuracy; α controls cluster count (α = 0.1 → 3 clusters, α = 5.0 → 10).
- `r/dirichlet_process_mixture.R` — `dirichletprocess::DirichletProcessGaussian`.

## When to use

- **Clustering with unknown K** — sidesteps AIC / BIC selection of K.
- **Overlapping subpopulations** in low-dimensional data (biomarkers, sensor readings, subgroups of a customer base).
- As a **flexible density estimator** — DP-Gaussian mixture is a nonparametric density.

## Alternatives

- **Truncated stick-breaking** (variational inference) — deterministic DP fit, much faster than Gibbs.
- **Pitman-Yor process** — heavier-tailed cluster-size distribution (power-law); good for language / text.
- **Sparse finite mixture** — set a large fixed `K` with a sparse Dirichlet prior on weights; often gives the same result more cheaply.

## Assumptions & caveats

- **Choice of α**: sensitive. Put a Gamma prior on α and sample it (Escobar-West 1995).
- **Label switching**: cluster labels are exchangeable; report cluster-based summary statistics (posterior similarity matrix, cluster sizes), not raw labels.
- **Convergence**: many iterations needed for very-mixed clusters; use multiple chains + Gelman-Rubin.
- **Small clusters**: DP often adds singleton clusters — a feature (heavy tails) or bug (overfitting) depending on interpretation. Post-process by merging singletons when reporting.

## Run

```
python techniques/dirichlet-process-mixture/python/dirichlet_process_mixture.py
Rscript techniques/dirichlet-process-mixture/r/dirichlet_process_mixture.R
```

**Refs:** Ferguson, T.S. "A Bayesian analysis of some nonparametric problems." *Ann. Stat.* 1(2), 209–230, 1973; Neal, R.M. "Markov chain sampling methods for Dirichlet process mixture models." *J. Comp. Graph. Stat.* 9(2), 249–265, 2000; Escobar, M.D. & West, M. "Bayesian density estimation and inference using mixtures." *JASA* 90(430), 577–588, 1995.

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

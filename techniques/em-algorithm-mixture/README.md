# EM Algorithm for Finite Mixtures (Reference §47.77)

Dempster, Laird & Rubin (1977). Iterate

    E-step: γ_ik = π_k f_k(x_i | θ_k) / Σ_j π_j f_j(x_i | θ_j)
    M-step: π_k = (1/n) Σ γ_ik, θ_k = weighted MLE with weights γ_ik.

Guaranteed monotone likelihood increase; converges to a local
optimum. Illustrated here on a 1-D Gaussian mixture.

## Files

- `python/em_algorithm_mixture.py` — from-scratch EM for a 1-D
  GMM + BIC model selection. Demo (n=2000, 3-component truth
  π=(0.40, 0.35, 0.25), μ=(−2.5, 0.5, 3.0), σ=(0.5, 0.8, 0.6)):
  - MLE π̂ = (0.40, 0.33, 0.27), μ̂ = (−2.51, 0.44, 2.95), σ̂ =
    (0.48, 0.78, 0.64) in 162 iterations.
  - BIC picks **K = 3** (7 773 vs 7 989 for K=2 and 7 786 for K=4).
- `r/em_algorithm_mixture.R` — `mclust::Mclust`,
  `mixtools::normalmixEM` (R); `sklearn.mixture.GaussianMixture`,
  from-scratch (Python).

## When to use

- **Latent-class / finite-mixture modelling** — subpopulation
  discovery, density estimation.
- **Missing data via data augmentation** — treat missing values
  as latent to iterate.
- **Hidden Markov models** — Baum-Welch is EM.
- **Factor analysis / probabilistic PCA** — closed-form E and M steps.

## When NOT to use

- **Non-identifiable mixtures** (label switching, weakly separated
  modes) — combine with post-hoc relabelling or Bayesian priors.
- **When ML surface is highly multi-modal** — multiple random
  starts or DA-EM (deterministic annealing).
- **When VI-style speed is enough** — variational inference sometimes
  scales better on massive data.

## Assumptions & caveats

- **Local optima** — try many starts; report the one with the best
  log-likelihood.
- **K selection** — BIC, AIC, ICL, or cross-validated log-likelihood.
- **Singularities** — a component may collapse onto a single point;
  guard σ_k with a small floor or use Bayesian priors.
- **Missing at random** for EM-on-missing-data validity.

## Related in this repo

- `gaussian-mixture-models`, `dirichlet-process-mixture`,
  `factor-mixture-model`, `latent-class-analysis`,
  `latent-profile-analysis`, `mixture-regression`,
  `mixture-of-experts` — model families sharing the EM machinery.
- `variational-inference`, `hmc-nuts`, `mcmc-metropolis-hastings`,
  `bayesian-hierarchical-models` — Bayesian alternatives.
- `hidden-markov-models` (via `state-space-models`), Baum-Welch
  is a special case.
- `hopkins-clusterability`, `silhouette-clusters`,
  `davies-bouldin-index`, `gap-statistic-cluster` — companion
  cluster-validation toolkit.

## Run

```
python techniques/em-algorithm-mixture/python/em_algorithm_mixture.py
Rscript techniques/em-algorithm-mixture/r/em_algorithm_mixture.R
```

**Refs:** Dempster, A.P., Laird, N.M. & Rubin, D.B. "Maximum likelihood from incomplete data via the EM algorithm." *JRSS-B* 39(1): 1-38, 1977; McLachlan, G.J. & Krishnan, T. *The EM Algorithm and Extensions*, 2nd ed., Wiley, 2008.

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

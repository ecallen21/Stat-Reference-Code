# Finite Mixture of Regressions (Reference §36.3)

DeSarbo & Cron (1988), Grün & Leisch (2007). Model heterogeneous
regression relationships as a `K`-component mixture:

```
y_i | z_i = k ~ N(x_i^T · β_k, σ_k²)
z_i           ~ Categorical(π_1, …, π_K)
```

Estimated by EM: **E-step** posterior class probabilities from
Gaussian densities; **M-step** weighted-OLS + per-class variance.

## When to use

- **Slope heterogeneity** with hidden subpopulations (segments of
  customers with different dose-response curves, market segments
  with distinct price sensitivities).
- **Latent-class regression** in marketing / psychology.

## When NOT to use

- **Observed groups** — plain group-by regression is transparent.
- **Very small classes** — EM local optima and label switching
  dominate.

## Files

- `python/mixture_regression.py` — from-scratch EM with weighted-
  OLS M-step. Demo (n=500, 2 classes with opposite ±1 slopes):
  **estimated slopes +0.95, −1.00; intercepts +0.49, −0.53;
  proportions 0.51 / 0.49**; MAP class recovery **90 %** vs truth.
- `r/mixture_regression.R` — `flexmix::flexmix` (canonical),
  `mixtools::regmixEM` (R); `sklearn.mixture` (Gaussian only) +
  custom EM (Python).

## Assumptions & caveats

- **Label switching** — EM may converge to a permutation of the
  true classes; align via posterior means or a reference class.
- **Class enumeration** — BIC selects `K`; over-fitting inflates
  entropy.
- **Local optima** — always multi-start (5-10 seeds).
- **Non-Gaussian residuals** — extend to mixtures of GLMs
  (`flexmix::FLXMRglm`).

## Related in this repo

- `gaussian-mixture-models`, `latent-class-analysis`,
  `latent-profile-analysis` — mixture cousins.
- `mixture-of-experts` — gating variables replace unconditional
  class weights.

## Run

```
python techniques/mixture-regression/python/mixture_regression.py
Rscript techniques/mixture-regression/r/mixture_regression.R
```

**Refs:** DeSarbo, W.S. & Cron, W.L. "A maximum likelihood methodology for clusterwise linear regression." *Journal of Classification*, 1988; Grün, B. & Leisch, F. "Fitting finite mixtures of generalized linear regressions in R." *Computational Statistics and Data Analysis*, 2007.

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

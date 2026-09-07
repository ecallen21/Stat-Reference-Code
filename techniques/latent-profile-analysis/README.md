# Latent Profile Analysis (Reference §36.2)

Gibson (1959), Vermunt & Magidson (2002). Continuous-indicator
cousin of Latent Class Analysis: a mixture of `K` multivariate
Gaussians on continuous items.

```
Y_i | class = k ~ MVN(μ_k, Σ_k)
class            ~ Categorical(π_1, …, π_K)
```

Estimated by EM; `K` chosen by **BIC / AIC / entropy** and
substantive plausibility.

## When to use

- **Person-centred typologies** — mental-health symptom clusters,
  personality profiles, patient-outcome trajectories.
- **Continuous survey / assessment items** where classes are more
  interpretable than a factor structure.

## When NOT to use

- **Binary / categorical items** — use LCA instead.
- **A single latent dimension** — factor analysis is simpler.
- **Class distinction driven by outliers** — refit with robust
  covariance or trim.

## Files

- `python/latent_profile_analysis.py` — diagonal-Σ Gaussian
  mixture via `sklearn` + BIC/AIC/entropy across `K ∈ {1, …, 5}`.
  Demo (n=600 with 3 true profiles): **BIC picks K=3**; recovered
  proportions (0.36, 0.41, 0.22) vs truth (0.40, 0.40, 0.20);
  means recovered to 2 dp.
- `r/latent_profile_analysis.R` — `mclust::Mclust`, `tidyLPA`,
  `MplusAutomation` (R); `sklearn.mixture.GaussianMixture`,
  `bnpy`, `pomegranate` (Python).

## Assumptions & caveats

- **Class enumeration** — fit K=1..K_max, compare BIC + inspect
  the profiles for substantive meaning; entropy > 0.80 usual
  cutoff for well-separated classes.
- **Local optima** — always multi-start (`n_init >= 5`).
- **Non-normal indicators** — LPA is a Gaussian mixture; skewed
  items may create spurious classes.
- **Class membership uncertainty** — report posterior probabilities
  alongside modal class assignment.

## Related in this repo

- `latent-class-analysis` — categorical-indicator cousin.
- `gaussian-mixture-models`, `latent-growth-mixture` — extended
  mixtures.
- `factor-mixture-model` — combines LPA with EFA.

## Run

```
python techniques/latent-profile-analysis/python/latent_profile_analysis.py
Rscript techniques/latent-profile-analysis/r/latent_profile_analysis.R
```

**Refs:** Gibson, W.A. "Three multivariate models: factor analysis, latent structure analysis, and latent profile analysis." *Psychometrika*, 1959; Vermunt, J.K. & Magidson, J. "Latent class cluster analysis." *Applied Latent Class Analysis*, 2002.

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

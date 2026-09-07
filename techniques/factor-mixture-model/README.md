# Factor Mixture Model (Reference §36.10)

Yung (1997), Lubke & Muthén (2005). Combines factor analysis
(continuous latent factor within class) with a categorical latent
class:

```
y_i | class = k, η_i ~ N(μ_k + Λ_k · η_i, Θ_k)
η_i                   ~ N(0, Ψ_k)
class                 ~ Categorical(π_1, …, π_K)
```

Motivation: even within a class, indicators may share common
latent factors. FMM combines LPA's between-class heterogeneity
with EFA's within-class dimensionality.

## When to use

- **Complex indicator structure** with both discrete typology and
  continuous dimensions.
- **Measurement invariance testing** across latent groups.
- **Personality / clinical typology research** where trait
  dimensions differ across profiles.

## When NOT to use

- **Pure factor model** — plain EFA / CFA is simpler.
- **Pure classes** — LPA / LCA is more parsimonious.
- **Small samples** — FMM has many parameters; unstable below
  ~500 subjects.

## Files

- `python/factor_mixture_model.py` — full-covariance Gaussian
  mixture + SVD extraction of loadings per class. Demo (n=800,
  p=6, 2 classes with disjoint loading patterns): recovered
  **π=(0.46, 0.54)** and **loadings within 0.05 of truth** for both
  classes.
- `r/factor_mixture_model.R` — `OpenMx`, `MplusAutomation`,
  `lavaan` (R); custom + `sklearn.mixture`, `semopy` (Python).

## Assumptions & caveats

- **Identification** — factor loadings and class means must jointly
  identify the model; usually fix one loading per factor to 1.
- **Local optima** — FMM is high-dimensional; multi-start EM
  essential.
- **Measurement invariance** — testing whether loadings are equal
  across classes is a common research question.
- **Approximation here** — the SVD-based recovery is illustrative;
  production FMM (Mplus, OpenMx) uses proper structural
  parameterisation with Λ, Θ, Ψ separately identified.

## Related in this repo

- `latent-profile-analysis`, `latent-class-analysis`,
  `mixture-regression` — mixture cousins.
- `factor-analysis`, `confirmatory-factor-analysis` (if present) —
  factor-only baselines.

## Run

```
python techniques/factor-mixture-model/python/factor_mixture_model.py
Rscript techniques/factor-mixture-model/r/factor_mixture_model.R
```

**Refs:** Yung, Y.-F. "Finite mixtures in confirmatory factor-analysis models." *Psychometrika*, 1997; Lubke, G.H. & Muthén, B. "Investigating population heterogeneity with factor mixture models." *Psychological Methods*, 2005.

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

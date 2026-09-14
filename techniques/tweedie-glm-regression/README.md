# Tweedie GLM (Reference §47.323)

Tweedie (1984); Jørgensen (1987). Exponential-dispersion
family with power variance function `Var(Y) = φ μ^p`:

| p     | Family                        |
|-------|-------------------------------|
| 0     | Normal (identity link)        |
| 1     | Poisson                       |
| (1,2) | **Compound Poisson-Gamma**    |
| 2     | Gamma                         |
| 3     | Inverse Gaussian              |

The p ∈ (1, 2) case has MASS at 0 and continuous positive tail —
standard for INSURANCE CLAIMS, rainfall, fisheries catch.

## Files

- `python/tweedie_glm_regression.py` — IRLS Tweedie GLM with
  p=1.5, log-link. Simulated compound Poisson-gamma data
  (n=1000, ~50% zeros); MLE recovers β̂ ≈ true β and dispersion
  φ̂ ≈ 0.89 (near true 1.0).
- `r/tweedie_glm_regression.R` — `statmod::tweedie`,
  `tweedie`, `cplm`, `mgcv` (R);
  `statsmodels.genmod.families.Tweedie`, from-scratch (Python).

## When to use

- **Insurance / actuarial** — claim amounts with excess
  zeros.
- **Rainfall / precipitation** — dry days + storm totals.
- **Fisheries** — many zero catches + positive skewed catches.
- **Any zero-inflated positive-continuous outcome**.

## When NOT to use

- **Pure counts** — use Poisson / NegBin.
- **Pure positive-continuous** — use Gamma / log-normal.
- **Structural vs sampling zeros** — hurdle / zero-inflated
  models offer sharper decomposition.

## Assumptions & caveats

- **p known or estimated** — p ∈ (1, 2) is the CP-Gamma
  region; profile-likelihood over p is standard.
- **Dispersion φ** — estimated by Pearson residuals or via
  the saddle-point log-likelihood (Dunn-Smyth 2005).
- **Link** — log-link most common; identity link risks
  negative μ.
- **Convergence** — IRLS may need step-halving under extreme
  skew.

## Related in this repo

- `hurdle-model`, `zero-inflated-regression`,
  `poisson-regression`, `negative-binomial-regression`,
  `gamma-regression`, `beta-regression` — companions.
- `dirichlet-regression`, `inverse-gaussian-glm` — other
  GLM families this batch.
- `gamlss` — distributional-regression generalisation.

## Run

```
python techniques/tweedie-glm-regression/python/tweedie_glm_regression.py
Rscript techniques/tweedie-glm-regression/r/tweedie_glm_regression.R
```

**Refs:** Tweedie, M.C.K. "An index which distinguishes between some important exponential families." In *Statistics: Applications and New Directions*, ISI, 1984; Jørgensen, B. "Exponential dispersion models." *JRSS-B*, 49(2): 127-162, 1987.

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

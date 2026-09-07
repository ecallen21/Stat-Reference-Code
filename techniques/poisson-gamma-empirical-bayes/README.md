# Poisson-Gamma Empirical Bayes (Reference §24.19)

Clayton & Kaldor (1987). Classic **shrinkage** estimator for
small-area disease rates / SMRs.

## Model

    Y_i | θ_i ~ Poisson(E_i · θ_i)              observed events
    θ_i       ~ Gamma(α, α / μ)                 site-specific risks

Posterior:

    θ_i | Y_i ~ Gamma(α + Y_i,  α/μ + E_i)
    Ê[θ_i | Y_i] = (α + Y_i) / (α/μ + E_i)

Empirical-Bayes: estimate `(μ, α)` by MoM or MLE on the marginal
negative-binomial and **plug in** for the site-level posterior mean.

## Effect

- Raw SMR = `Y_i / E_i` has enormous variance for small `E_i`.
- The shrunken estimate pulls toward the overall mean `μ` in
  proportion to how weak the evidence is (small `E_i`).

## Files

- `python/poisson_gamma_empirical_bayes.py` — MoM + MLE for `(μ, α)`,
  closed-form Gamma posterior mean. Demo (n=200 areas, true α=5.0,
  μ=1.0): estimated (0.98, 3.94); shrunken MAE 0.229 vs raw 0.344
  (34 % reduction); in the small-area subset (bottom 25 % of E),
  shrinkage cuts MAE 0.583 → 0.298 (49 %).
- `r/poisson_gamma_empirical_bayes.R` — `DCluster::empbaysmooth`,
  `SpatialEpi::eBayes`, `INLA` (R); `pymc`, `statsmodels`
  NegativeBinomial (Python).

## When to use

- **Disease mapping** — small-area SMR / IRR mapping in
  cancer / infection surveillance.
- **Any Poisson rate with lots of low-count strata** — hospital
  quality metrics, insurance claim rates, sports goals.
- **Simple, defensible shrinkage** — when a full spatial model
  (BYM, CAR) is too heavy.

## When NOT to use

- **Strong spatial correlation** — Poisson-gamma has no neighbour
  information; use BYM / BYM2 (`INLA`) instead.
- **Zero-inflation** — a large excess-zero mass breaks the NB fit;
  use ZIP / ZINB.
- **Rare covariate adjustment needed** — extend to a Poisson-GLM
  with random intercept.

## Assumptions & caveats

- **Exchangeability** — sites are treated as interchangeable draws
  from one Gamma; violated by regional structure.
- **Correctly specified `E_i`** — expected counts from
  age-standardisation must be reliable.
- **Uncertainty** — plug-in EB understates uncertainty (ignores
  variance in `(μ, α)`); use bootstrap or a full Bayes for CIs.
- **Extreme sites** — a genuinely outlying site is pulled toward the
  centre by construction; report both raw and shrunken.

## Related in this repo

- `bayesian-hierarchical-models`, `conditional-autoregressive-car`
  — spatial hierarchical alternatives.
- `james-stein-shrinkage` — Gaussian analogue.
- `negative-binomial-regression` — the marginal model here.
- `spatial-glm`, `spatial-scan-cluster` — disease-mapping siblings.

## Run

```
python techniques/poisson-gamma-empirical-bayes/python/poisson_gamma_empirical_bayes.py
Rscript techniques/poisson-gamma-empirical-bayes/r/poisson_gamma_empirical_bayes.R
```

**Refs:** Clayton, D. & Kaldor, J. "Empirical Bayes estimates of age-standardized relative risks for use in disease mapping." *Biometrics*, 43(3): 671-681, 1987; Marshall, R.J. "Mapping disease and mortality rates using empirical Bayes estimators." *Applied Statistics*, 40(2): 283-294, 1991.

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

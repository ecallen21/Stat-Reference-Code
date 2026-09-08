# Weighted Quantile Sum (WQS) Regression (Reference §47.62)

Carrico, Gennings, Wheeler & Factor-Litvak (2015). Regresses an
outcome on a **weighted index** of quantile-scored exposures:

    y = β₀ + β₁ · (Σᵢ wᵢ · qᵢ(xᵢ)) + Z γ + ε,     Σᵢ wᵢ = 1,  wᵢ ≥ 0.

Simultaneously estimates the mixture EFFECT `β₁` and each
exposure's SHARE `wᵢ`. Directional constraint (positive or
negative) prevents sign cancellation across highly correlated
exposures — a standard problem in environmental epi.

## Files

- `python/weighted_quantile_sum_wqs.py` — from-scratch WQS with
  `SLSQP` for the constrained weight optimisation + bootstrap
  train/validation splits. Demo (n=400, p=6 correlated exposures,
  truly bad at indices 0 and 3):
  - 80/80 bootstrap replicates in the positive direction
  - w̄[0] = 0.44, w̄[3] = 0.45, all other w̄ ≤ 0.05
  - β̂₁ = 0.459 vs truth 0.400.
- `r/weighted_quantile_sum_wqs.R` — `gWQS`,
  `groupWQS` (R); `wqspy`, from-scratch (Python).

## When to use

- **Environmental / occupational exposure mixtures** —
  co-pollutants, pesticide panels, PFAS.
- **Nutrient / biomarker composites** — vitamins, oxidative-
  stress markers.
- **Chemical safety** — regulatory endpoints from EDC panels.

## When NOT to use

- **Uncorrelated exposures** — plain multiple regression is
  simpler and unbiased.
- **Bidirectional effects in the same mixture** — the directional
  constraint kills sign-cancellation; use qgcomp instead when both
  directions are plausible.
- **Nonlinear or interaction-dominant mixtures** — use Bayesian
  kernel machine regression (BKMR) or random-forest-based methods.

## Assumptions & caveats

- **Direction assumption** — pick + or −; wrong direction filters
  out the truth.
- **Quantile choice** — 4 (quartiles) is the standard; sensitivity
  to n_quant should be reported.
- **Train/validation split** — key for weight identifiability;
  Carrico recommends 40/60 with B ≥ 100 bootstraps.
- **Standard errors** — bootstrap-based; sandwich estimators are
  not straightforward under the simplex constraint.
- **Modern alternative**: qgcomp (Keil et al 2020) drops the
  bootstrap and handles bidirectional mixtures.

## Related in this repo

- `mixture-regression`, `dirichlet-process-mixture` — mixture-
  modelling cousins.
- `group-lasso`, `sparse-pca`, `ridge-lasso-elasticnet` —
  correlated-feature regularisation.
- `mediation-analysis`, `causal-forest`, `dml-double-ml` — related
  effect-estimation frameworks.
- `bayesian-hierarchical-models`, `bayesian-glms` — BKMR analog.

## Run

```
python techniques/weighted-quantile-sum-wqs/python/weighted_quantile_sum_wqs.py
Rscript techniques/weighted-quantile-sum-wqs/r/weighted_quantile_sum_wqs.R
```

**Refs:** Carrico, C., Gennings, C., Wheeler, D.C. & Factor-Litvak, P. "Characterization of weighted quantile sum regression for highly correlated data in a risk analysis setting." *J Agric Biol Environ Stat* 20(1): 100-120, 2015; Keil, A.P. et al. "A quantile-based g-computation approach to addressing the effects of exposure mixtures." *Environ Health Perspect* 128(4): 047004, 2020.

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

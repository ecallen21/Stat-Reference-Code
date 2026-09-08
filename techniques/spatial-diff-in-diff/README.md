# Spatial Diff-in-Differences (Reference §15.43)

Delgado & Florax (2015); Chagas et al. (2016). Extends DiD when
treatment ASSIGNMENT spills over across neighboring units.

## SLX-DiD specification

    y_it = α + β·D_i + γ·Post_t + δ·D_i·Post_t + θ·(W D_i)·Post_t + ε_it

- **δ** — direct effect (treated units, post period).
- **θ** — spillover onto untreated neighbours of treated units.
- **W** — row-normalised spatial weights matrix.

Alternative: SAR-DiD adds `ρ · W · y_it` on the RHS.

## Files

- `python/spatial_diff_in_diff.py` — SLX-DiD via OLS + block-
  diagonal `W_obs` from scratch. Demo (N=40 on a line, treated
  cluster units 15-24, true δ=2.0, θ=0.8): recovers δ̂ = +2.38,
  θ̂ = +0.62.
- `r/spatial_diff_in_diff.R` — `spatialreg + spdep`, `plm + spml`,
  `splm`, `fixest::feols` (R); PySAL, `linearmodels`, from-scratch
  (Python).

## When to use

- **Local-level interventions** — schools, health facilities, policy
  zones — where neighbouring units are affected.
- **Rollout with contiguous treatment groups** — regional programmes.
- **Testing SUTVA-violation** — SLX-DiD explicitly quantifies
  spillover.

## When NOT to use

- **No plausible spatial spillover** — plain DiD suffices.
- **Weights matrix unclear** — sensitivity to W definition can
  overwhelm treatment estimation.
- **Very small T** — need pre + post per unit; more pre-periods
  strengthen parallel-trends checks.

## Assumptions & caveats

- **Weights matrix W** — nearest-neighbour, k-nearest, kernel-
  distance; report sensitivity.
- **Parallel trends** — required as in classical DiD.
- **Endogenous SAR term** requires ML / GMM (spatialreg); OLS
  works for SLX but not SAR.
- **Cluster / spatial-HAC SE** — errors are spatially correlated;
  use `spdep::listw2mat` + Conley SE or wild-cluster bootstrap.

## Related in this repo

- `diff-in-diff`, `staggered-did`, `synthetic-did`, `event-study`
  — DiD family.
- `spatial-autoregressive-sar`, `conditional-autoregressive-car`,
  `morans-i-gearys-c`, `spatial-scan-cluster` — spatial siblings.
- `iv-2sls`, `iptw` — non-spatial causal cousins.

## Run

```
python techniques/spatial-diff-in-diff/python/spatial_diff_in_diff.py
Rscript techniques/spatial-diff-in-diff/r/spatial_diff_in_diff.R
```

**Refs:** Delgado, M.S. & Florax, R.J. "Difference-in-differences techniques for spatial data: local autocorrelation and spatial interaction." *Economics Letters*, 137: 123-126, 2015; Chagas, A.L. et al. "A spatial difference-in-differences analysis of the impact of sugarcane production on respiratory diseases." *PLOS ONE*, 11(3): e0150989, 2016.

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

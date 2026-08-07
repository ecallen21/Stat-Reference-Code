# Synthetic Control (Reference §15.10)

Single treated unit, many potential controls (the "donor pool"), and a long pre-treatment time series. Construct a **weighted average** of donors that matches the treated unit's pre-treatment trajectory; use the same weighted average as the counterfactual after treatment.

## Optimization

Choose non-negative weights `W = (w_1, ..., w_J)` summing to 1 that minimize the pre-treatment fit:

```
min_W   Σ_t (Y_{1, t}^pre − Σ_j w_j Y_{j, t}^pre)²
s.t.    w_j ≥ 0,   Σ_j w_j = 1
```

Optional: weight time periods by `V` for pre-treatment covariate balance (`Synth::synth`).

## Effect

```
ATT_t = Y_{1, t}^post − Σ_j w_j Y_{j, t}^post
```

Report the time series of gaps; often summarize as an average post-period ATT.

## Inference: placebo test

Reassign the "treatment" to each donor in turn; compute the ratio `post-RMSE / pre-RMSE` for each. Rank the true treated unit's ratio among all placebos. A ratio in the top 5% supports significance.

## Files

- `python/synthetic_control.py` — SLSQP optimization for the simplex-constrained weights + gap trajectory + pre/post RMSE. Demo (20 donors, 20 pre + 10 post periods, true ATT = 3): recovers avg post-ATT = 2.72; pre-period RMSE 0.18 (well-matched); post-period gaps jump at t = 20.
- `r/synthetic_control.R` — pointer to `Synth::synth` (canonical Abadie-Diamond-Hainmueller) or `gsynth::gsynth` (with covariates and multiple treated units).

## When to use

- **Comparative case studies** — one region / firm / country adopts a policy; you have long pre-treatment data and many similar untreated units.
- **Effect of a discrete event** — natural disaster, tax reform, sanctions, product launch.
- **Small-N settings** where standard DID lacks a good comparison group.

## When NOT to use

- **No good donor pool** — the treated unit is unlike any control; the weighted average falls outside the convex hull of donors.
- **Very short pre-period** — insufficient signal to identify weights.
- **Post-treatment shocks to donors** — spillovers, contamination, or macro shocks common to all units.

## Extensions

- **Multiple treated units** — average the individual synthetic controls, or use `gsynth`.
- **Covariates** — include pre-treatment predictors in the matching objective with weights `V`.
- **Generalized SC** — Xu (2017) `gsynth` uses a factor model when donor pool is not convex-hull-sufficient.
- **Augmented SC** — Ben-Michael et al. (2021) combines SC with an outcome regression.
- **Bayesian / matrix-completion SC** — recent methods with fully probabilistic uncertainty.

## Assumptions & caveats

- **Convex-hull assumption**: treated pre-period must lie inside donor convex hull for interpolation to make sense.
- **No anticipation** or spillover of the treatment to donors.
- **Weights are point estimates**; SEs from placebo distribution are the standard inference, not analytic.

## Run

```
python techniques/synthetic-control/python/synthetic_control.py
Rscript techniques/synthetic-control/r/synthetic_control.R
```

**Refs:** Abadie, A. & Gardeazabal, J. "The economic costs of conflict: a case study of the Basque Country." *AER* 93(1), 113–132, 2003; Abadie, A., Diamond, A. & Hainmueller, J. "Synthetic control methods for comparative case studies." *JASA* 105(490), 493–505, 2010; Abadie, A. "Using synthetic controls: feasibility, data requirements, and methodological aspects." *J. Econ. Lit.* 59(2), 391–425, 2021.

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

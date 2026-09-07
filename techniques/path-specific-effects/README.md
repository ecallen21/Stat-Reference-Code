# Path-Specific Effects (Reference §15.41)

Avin, Shpitser & Pearl (2005); VanderWeele & Chiba (2014). Extends
natural direct/indirect effects to DAGs with **multiple mediators**,
decomposing the total effect along specific causal pathways.

## Decomposition (two ordered mediators)

For `T → M₁ → M₂ → Y` (with `T → M₂`, `T → Y`, `M₁ → Y`, `M₂ → Y`):

    TE = PSE_direct
       + PSE_{T → M₁ → Y}
       + PSE_{T → M₂ → Y}          (independent of M₁)
       + PSE_{T → M₁ → M₂ → Y}     (chain)

For a **linear-Gaussian SCM**, Wright's path coefficients give
each PSE as a product of edge coefficients.

## Files

- `python/path_specific_effects.py` — closed-form linear-SCM path
  decomposition from OLS regressions of M₁, M₂, Y from scratch.
  Demo (n=6000, true α₁=0.5, α₂=0.3, μ₁=0.6, β₁=0.4, γ₁=0.7,
  γ₂=0.5): recovers direct +0.40, via M₁ +0.37 (truth 0.35), via M₂
  +0.14 (truth 0.15), chain +0.14 (truth 0.15), total +1.05 (truth
  1.05).
- `r/path_specific_effects.R` — `paths::paths` (Zhou-Yamamoto),
  `medflex::neImpute`, `lavaan::sem` (R); DoWhy, causallib (Python).

## When to use

- **Multi-mediator mechanism decomposition** — separate the role of
  each mediator and their chain.
- **Policy-lever analysis** — which pathway to intervene on for the
  biggest reduction in TE.
- **Cross-world identification with a graph** — Avin-Shpitser-Pearl
  identifiability tells you which PSEs are identified at all.

## When NOT to use

- **Not all PSEs identified** — the recanting-witness condition
  (some paths are cross-world unidentified without extra structure).
  DoWhy / paths raise an error in that case.
- **Nonlinear interactions between mediators** — the linear-SCM
  formulas break; use imputation-based PSE estimators.
- **Single mediator** — use the simpler NDE/NIE from
  `mediation-natural-effects`.

## Assumptions & caveats

- **DAG is correct** — the whole decomposition rests on the causal
  graph you posit.
- **Sequential ignorability** at every stage (T ⊥ Ms; Ms ⊥ Y given
  the past).
- **Cross-world independence** — needed to identify certain PSEs;
  untestable and controversial.
- **Bootstrap SEs** — path products have complicated variances;
  analytic delta-method exists but bootstrap is standard.

## Related in this repo

- `mediation-natural-effects` — single-mediator NDE / NIE.
- `mediation-analysis`, `multilevel-mediation` — classical
  alternatives.
- `causal-discovery-pc` — how to learn the DAG on which PSEs live.
- `path-analysis`, `cfa-confirmatory-factor` — SEM cousins.

## Run

```
python techniques/path-specific-effects/python/path_specific_effects.py
Rscript techniques/path-specific-effects/r/path_specific_effects.R
```

**Refs:** Avin, C., Shpitser, I. & Pearl, J. "Identifiability of path-specific effects." *IJCAI*, 2005; VanderWeele, T.J. & Chiba, Y. "Multiple mediators: sensitivity analysis for path-specific effects." *Epidemiology*, 25(1): 145-146, 2014.

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

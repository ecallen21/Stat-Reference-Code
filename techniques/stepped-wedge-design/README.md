# Stepped-Wedge Cluster Trial (Reference §47.265)

Hussey & Hughes (2007). Clusters are randomised to the TIMING
of intervention rollout. All clusters end up treated; the
design gives every cluster its own control period, gaining
efficiency over parallel CRT:

```
Y_{ijt} = μ + β_t + θ · X_{ijt} + u_i + e_{ijt}
u_i ~ N(0, τ²)          (cluster random effect)
```

`θ` is the intervention effect; analyse via mixed-effects
regression or GEE.

## Files

- `python/stepped_wedge_design.py` — simulator + design
  matrix visualisation + cluster+period FE OLS analysis.
  Demo: K=12, T=5, m=10, true θ=0.5, τ=0.3. Cluster+period
  FE OLS recovers θ̂ close to truth (200-trial mean = 0.52,
  SE = 0.14). Naive analysis without time adjustment gives
  θ̂ = 0.62 (biased upward by secular trend baked into the
  simulator).
- `r/stepped_wedge_design.R` — `swCRTdesign::swPwr`,
  `SWSamp`, `lmerTest::lmer`, `geepack::geeglm` (R);
  `statsmodels.MixedLM`, `linearmodels.PanelOLS`, from-scratch
  (Python).

## When to use

- **Universal-adoption interventions** — new guideline,
  training programme, or IT system that must roll out to all
  clusters eventually.
- **Fewer clusters, more time** — SW is more efficient than
  parallel CRT with the same number of clusters.
- **Practical constraints** — logistics forces staggered
  implementation anyway.

## When NOT to use

- **Time-limited effect** — SW mixes early and late exposure;
  a pure parallel CRT is cleaner if effect decays.
- **Strong secular trend** with few periods — period effects
  become unstable.
- **Highly heterogeneous cluster response** — the standard
  Hussey-Hughes analysis assumes constant intervention effect
  across time.

## Assumptions & caveats

- **Time trend adjustment** — MUST include period fixed
  effects; the naive analysis is severely confounded (see
  demo).
- **Constant treatment effect** — Kenny et al 2021 showed
  bias if the effect varies over exposure time; use
  exposure-time-varying models (ETI) for robustness.
- **Autocorrelation within cluster** — Hooper et al 2016
  recommend exchangeable-decaying correlation, not simple
  exchangeable, when periods are long.
- **Design effect** — larger than parallel CRT with the
  same number of clusters when ICC is moderate.

## Related in this repo

- `cluster-randomized-trial` — parallel CRT alternative.
- `staggered-did` — related time-varying design in
  econometrics.
- `factorial-2x2-trial` — factorial design cousin.

## Run

```
python techniques/stepped-wedge-design/python/stepped_wedge_design.py
Rscript techniques/stepped-wedge-design/r/stepped_wedge_design.R
```

**Refs:** Hussey, M.A. and Hughes, J.P. "Design and analysis of stepped wedge cluster randomized trials." *Contemp. Clin. Trials*, 28(2): 182-191, 2007.

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

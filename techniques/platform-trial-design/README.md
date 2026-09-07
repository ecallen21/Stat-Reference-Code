# Platform / Master-Protocol Trial Design (Reference §44.14)

Woodcock & LaVange (2017); Berry et al. (2015). Umbrella / basket /
platform trials share a common infrastructure to compare multiple
therapies against a common control, adding arms as they become
available and dropping them for futility.

## Terminology

| Design | Structure |
|---|---|
| **Umbrella** | One disease, many treatments (biomarker subgroups) |
| **Basket** | Many diseases (tumor histologies), one target therapy |
| **Platform** | Open-ended: arms enter / exit dynamically; shared control |

## Design features

- Bayesian posterior-based dose comparison against pooled control.
- Interim futility / graduation rules on `P(θ_arm > θ_ctrl | data)`.
- Adaptive allocation (Thompson-lite between active arms).
- Shared control reduces sample-size vs traditional 1:1 trials.

## Files

- `python/platform_trial_design.py` — simulates a two-arm-plus
  platform trial with graduation (P ≥ 0.975) and futility (P ≤ 0.10)
  posterior thresholds. Demo (200 simulations, control 0.30, A=0.30
  null, B=0.42 winner, C=0.25 harmful): winner B graduates 93%,
  harmful C hit for futility 71%, null A 36% futile / 56% inconclusive.
- `r/platform_trial_design.R` — `adaptr`, `pipe`, `MAMS`, `BOP2`,
  `octopus`, `gsDesign` (R); from-scratch (Python).

## When to use

- **Rare / heterogeneous diseases** — oncology (I-SPY2, GBM AGILE),
  Alzheimer's (DIAN-TU), COVID-19 (RECOVERY, REMAP-CAP).
- **Many candidate therapies** — pipeline is deeper than a single
  trial can accommodate.
- **Learning trial infrastructure** — share IRB / consent / control
  across substudies.
- **Adaptive dose finding** — combine multiple doses with
  graduation criteria.

## When NOT to use

- **Single-comparison Phase III with strict frequentist type-I** —
  regulatory hurdle for platform-wide error control.
- **Small pipeline** — the setup / operational cost outweighs
  benefits.
- **Absent statistical infrastructure** — needs a DSMB, coordinating
  centre, and sophisticated simulation for operating
  characteristics.

## Assumptions & caveats

- **Shared control comparability across time** — check for
  temporal drift; use time-strata or non-concurrent-control
  adjustments (Lee & Wason 2020).
- **Multiplicity control** — Bayesian graduation thresholds do not
  automatically control family-wise error; pre-specify and simulate.
- **Adaptive-allocation bias** — early winners get more patients;
  Bayesian estimators handle this correctly, but frequentist ones
  need adjustment.
- **Operational complexity** — real platforms take years of design;
  simulate operating characteristics extensively.

## Related in this repo

- `response-adaptive-randomization`, `bayesian-ab-testing`,
  `multi-armed-bandits` — allocation cousins.
- `sequential-analysis`, `always-valid-inference` — sequential
  monitoring.
- `mde-sample-size`, `sample-size` (see design-of-experiments
  cluster) — fixed-design counterparts.
- `basket / umbrella` trial-specific tooling in `pipe` / `BOP2`.

## Run

```
python techniques/platform-trial-design/python/platform_trial_design.py
Rscript techniques/platform-trial-design/r/platform_trial_design.R
```

**Refs:** Woodcock, J. & LaVange, L.M. "Master protocols to study multiple therapies, multiple diseases, or both." *NEJM*, 377(1): 62-70, 2017; Berry, S.M. et al. "The platform trial: an efficient strategy for evaluating multiple treatments." *JAMA*, 313(16): 1619-1620, 2015.

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

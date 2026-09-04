# Interference + Cluster Randomization (Reference §44.6)

Blake & Coey (2014), Saveski et al. (2017). SUTVA fails when
treatment on one unit affects other units — marketplaces, social
networks, ride-sharing. Two design fixes:

- **Cluster randomisation** — randomise entire clusters (city,
  ride-share zone, social community). Analysis: cluster-mean
  two-sample test; adjust SE by the **design effect**
  `1 + (m − 1) · ICC`.
- **Switchback / time-based** — alternate treatment across time
  slots within the same market so each unit contributes both
  control and treatment periods.

## When to use

- **Marketplace / two-sided platform** — treatment of a rider
  affects driver supply.
- **Social-network** experiments — treatment spills to friends.
- **Referral / viral** features.

## When NOT to use

- **User-independent products** where SUTVA plausibly holds — a
  standard user-randomised test is fine.

## Files

- `python/interference_cluster.py` — cluster-mean t-test vs naive
  per-user t. Demo (20 clusters × 50 users, cluster-randomised
  effect = 0.20, unmeasured cluster noise σ=0.6): **naive p =
  3 × 10⁻²⁷** (spuriously certain) vs **cluster p = 0.080**
  (honest); estimated ICC = 0.76 → design effect ~38.
- `r/interference_cluster.R` — `inferference`, `DeclareDesign`,
  `clusterSEs`, `lme4` (R); `linearmodels`, `statsmodels`
  (cluster-robust SEs), custom (Python).

## Assumptions & caveats

- **Randomisation unit = analysis unit** — never analyse a
  cluster-randomised experiment per-user without accounting for
  the design.
- **Design effect** requires an ICC estimate; use pilot data.
- **Switchback windows** must be long enough to reach steady state
  after each switch.
- **Ego-network / cluster-aware analysis** (Saveski 2017) handles
  partial interference across clusters.

## Related in this repo

- `hierarchical-models`, `generalized-linear-mixed-models` — the
  formal statistical analogue.
- `ab-test-fundamentals`, `mde-sample-size` — the plain designs
  cluster experiments replace.

## Run

```
python techniques/interference-cluster/python/interference_cluster.py
Rscript techniques/interference-cluster/r/interference_cluster.R
```

**Refs:** Blake, T. & Coey, D. "Why marketplace experimentation is harder than it seems." *EC*, 2014; Saveski, M., Pouget-Abadie, J., Saint-Jacques, G., Duan, W., Ghosh, S., Xu, Y., & Airoldi, E.M. "Detecting network effects: randomizing over randomized experiments." *KDD*, 2017.

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

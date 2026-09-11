# Cluster-Randomized Trial (Reference §47.266)

Donner & Klar (2000). Randomise CLUSTERS (clinics, schools,
villages) rather than individuals. Within-cluster correlation
(ICC = ρ) inflates required sample size by the DESIGN EFFECT:

```
DEFF = 1 + (m − 1) · ρ
```

where `m` is average cluster size. Effective `n = N / DEFF`.
Analysis via mixed-effects, GEE, or cluster-summary t-test.

## Files

- `python/cluster_randomized_trial.py` — DEFF formula, SS
  calculator, simulator + cluster-summary t-test. Demo:
  δ=0.3, σ=1.0 gives individually-randomised n=175/arm. At
  m=30, ICC=0.05: DEFF=2.45, need 15 clusters per arm. At
  m=100, ICC=0.20: DEFF=20.8, need 37 clusters per arm.
  Simulated CRT (k=20, m=30, ICC=0.05): cluster-summary
  t=2.45 (p=0.019); naive individual-level t=3.87 (p=0.0001)
  — the naive test is anti-conservative. Empirical ICC=0.061.
- `r/cluster_randomized_trial.R` — `clusterPower`, `CRTSize`,
  `lmerTest::lmer`, `geepack::geeglm`, `ICCbin` (R);
  `statsmodels.MixedLM`, `linearmodels.PanelOLS`, from-scratch
  (Python).

## When to use

- **Interventions delivered at cluster level** — school-based
  programmes, primary-care practice interventions.
- **Contamination risk** in individual randomisation — cluster
  level prevents spillover.
- **Administrative constraints** — hard to randomise
  individuals within a shared setting.

## When NOT to use

- **Cluster very homogeneous** (ρ near 0) — no efficiency
  penalty, but no need for cluster analysis either.
- **Few clusters (< 10 per arm)** — small-sample corrections
  become essential; consider augmented sampling.
- **Contamination is minimal** — individually randomised
  designs are more efficient.

## Assumptions & caveats

- **ICC estimation** — publish ICC estimates from your own
  data for future trial planning.
- **Cluster imbalance** — unequal sizes reduce power; use
  variance-inflated `DEFF = 1 + (CV² + 1)(m̄ − 1) ρ`.
- **Analysis choice** — cluster-summary t-test (Donner-Klar)
  is exact for small `k`; GEE and MixedLM need at least ~ 40
  clusters for reliable SE.
- **Ignoring clustering** anti-conservatively inflates
  Type-I — never analyse individual observations as if
  independent.

## Related in this repo

- `stepped-wedge-design` — the time-varying cluster cousin.
- `gee` — analysis workhorse for CRTs.
- `mde-sample-size` — the individual-randomisation formula
  DEFF adjusts.
- `wild-cluster-bootstrap` — small-`k` inference correction.

## Run

```
python techniques/cluster-randomized-trial/python/cluster_randomized_trial.py
Rscript techniques/cluster-randomized-trial/r/cluster_randomized_trial.R
```

**Refs:** Donner, A. and Klar, N. *Design and Analysis of Cluster Randomization Trials in Health Research*, Arnold Publishers, London, 2000.

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

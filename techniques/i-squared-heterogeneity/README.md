# I² Heterogeneity (Reference §47.272)

Higgins & Thompson (2002); Higgins et al (2003). Percentage of
total variability in a meta-analysis attributable to
BETWEEN-STUDY heterogeneity:

```
I² = max(0, (Q − df) / Q) · 100%
H² = Q / df
τ² = between-study variance (DL estimator)
```

Cochrane thresholds: I² 25% low, 50% moderate, 75% substantial,
> 75% considerable. Complementary to τ² (absolute) and Q
(hypothesis test).

## Files

- `python/i_squared_heterogeneity.py` — Q + I² + H² + τ² +
  prediction interval. Demo across three heterogeneity levels:
  homogeneous τ=0 gives I²≈0%; moderate τ=0.2 gives I²≈47%;
  high τ=0.4 gives I²≈96%. 95% prediction interval for a new
  study widens dramatically with τ (from ≈(0.25, 0.75) to
  ≈(−0.34, 1.34)).
- `r/i_squared_heterogeneity.R` — `metafor::rma`,
  `meta::metagen`, `dmetar::pcurve` (R); PythonMeta,
  from-scratch (Python).

## When to use

- **Any random-effects meta-analysis** — always report I², τ²
  and the Q test.
- **Deciding FE vs RE** — high I² tilts toward RE + HKSJ.
- **Communicating uncertainty** — the prediction interval is
  MUCH more honest than the CI of the pooled mean when
  τ² > 0.

## When NOT to use

- **Very small K** — I² is noisy and can be misleading at
  K < 5.
- **As sole diagnostic** — I² depends on within-study
  precision; report τ² alongside for absolute scale.
- **Cross-scale comparisons** — I² depends on the effect
  scale (OR vs RR) and cannot be compared across meta-analyses
  with different scales.

## Assumptions & caveats

- **DL vs REML τ²** — I² depends on τ² estimator;
  REML-based I² is now preferred.
- **Confidence interval for I²** — use non-central-χ² or Q
  profile intervals (metafor::confint reports both).
- **I² floor at 0** — when Q < df, I² is set to 0; a nominal
  I² can therefore hide small heterogeneity.
- **Prediction interval** requires K > 2 and finite df.

## Related in this repo

- `dersimonian-laird-random-effects` — the pooling framework.
- `hartung-knapp-sidik-jonkman` — companion CI adjustment.
- `leave-one-out-meta`, `cumulative-meta-analysis` — study-
  level sensitivity analyses.
- `meta-regression`, `meta-analysis` — the containing
  framework.

## Run

```
python techniques/i-squared-heterogeneity/python/i_squared_heterogeneity.py
Rscript techniques/i-squared-heterogeneity/r/i_squared_heterogeneity.R
```

**Refs:** Higgins, J.P.T. and Thompson, S.G. "Quantifying heterogeneity in a meta-analysis." *Stat. Med.*, 21(11): 1539-1558, 2002; Higgins, J.P.T. et al. "Measuring inconsistency in meta-analyses." *BMJ*, 327(7414): 557-560, 2003.

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

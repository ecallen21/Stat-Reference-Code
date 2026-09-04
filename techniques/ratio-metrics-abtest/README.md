# Ratio Metrics + Delta Method for A/B Tests (Reference §44.10)

Deng, Knoblich & Lu (2018). Many business metrics are **ratios**
of two random variables — revenue-per-user, CTR = clicks /
impressions, orders-per-session. Naive per-user t-test on the ratio
is invalid when:

- Unit of randomisation ≠ ratio denominator (e.g., user-randomised
  but CTR aggregated over impressions).
- Numerator and denominator are correlated.

## Delta method

For a ratio `R = Y / X`:

```
Var(R̂) ≈ (μ_y / μ_x)² · [ Var(Y)/μ_y² − 2·Cov(Y, X)/(μ_y·μ_x)
                            + Var(X)/μ_x² ]  / n
```

Diff-of-ratios SE = √(Var_C + Var_T).

## When to use

- **Every ratio metric** in an A/B test.
- **Impression-level CTR** with user-level randomisation.
- **Session / user-conversion** rates.

## When NOT to use

- **Ratio denominator is deterministic** (session count fixed by
  design) — direct variance of the numerator is enough.
- **Ratio very small** or `μ_x ≈ 0` — delta-method Taylor
  expansion breaks down; bootstrap instead.

## Files

- `python/ratio_metrics_abtest.py` — delta-method SE for a
  difference of ratios (custom). Demo (n=5000, imps~Poisson(20),
  CTR 5.0 % → 5.5 %): pooled ratio C=0.0501, T=0.0546; **abs
  lift 0.0045, delta-SE 0.0010, 95 % CI (0.0026, 0.0065)** —
  per-user mean CTR is a different quantity than pooled ratio.
- `r/ratio_metrics_abtest.R` — `msm::deltamethod`, `boot`,
  `sandwich` (R); custom + `scipy.stats` (Python).

## Assumptions & caveats

- **Bootstrap alternative** — non-parametric, no delta-method
  Taylor assumption; slower but robust for skewed metrics.
- **Cluster-robust variance** — cluster-randomised ratio metrics
  need clustered SEs on top of the delta method.
- **Report the pooled ratio, not per-user mean** — they are
  different estimands.
- **Winsorise heavy tails** in the numerator before applying delta.

## Related in this repo

- `delta-method` — general delta method (this is the A/B special
  case).
- `ab-test-fundamentals`, `cuped-variance-reduction` — companion
  A/B tools.
- `bca-bootstrap` — bootstrap alternative for ratio SEs.

## Run

```
python techniques/ratio-metrics-abtest/python/ratio_metrics_abtest.py
Rscript techniques/ratio-metrics-abtest/r/ratio_metrics_abtest.R
```

**Refs:** Deng, A., Knoblich, U., & Lu, J. "Applying the delta method in metric analytics: a practical guide with novel ideas." *KDD*, 2018.

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

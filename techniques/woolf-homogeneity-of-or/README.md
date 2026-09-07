# Woolf Test of OR Homogeneity (Reference §4.17)

Woolf (1955). Tests whether K stratum-specific odds ratios are
**homogeneous** — a prerequisite for reporting a single pooled OR
(Mantel-Haenszel / Cochran-Mantel-Haenszel).

## Statistic

    X_W = Σ_k w_k · (log OR_k − log OR_pooled)²   ~   χ²_{K − 1}
    w_k = 1 / Var(log OR_k)   with   Var = 1/a + 1/b + 1/c + 1/d

Reject `H₀: OR₁ = … = OR_K` if `X_W` exceeds the chi-square critical
value.

## Files

- `python/woolf_homogeneity_of_or.py` — Woolf statistic with
  Haldane cell-continuity correction from scratch. Demo (3 strata):
  homogeneous ORs (1.71, 2.0, 1.86) → X = 0.12, p = 0.94 (do not
  reject); heterogeneous ORs (4.0, 1.0, 0.25) → X = 25.6, p =
  2.7e-6 (strongly reject → do not pool).
- `r/woolf_homogeneity_of_or.R` — `DescTools::WoolfTest`,
  `metafor::rma.mh`, `epitools`, `stats::mantelhaen.test` (R);
  `statsmodels.stats.contingency_tables.StratifiedTable`,
  from-scratch (Python).

## When to use

- **Multi-stratum 2 × 2 tables** — before pooling with Mantel-
  Haenszel or CMH.
- **Test of effect modification** — heterogeneous OR = evidence of
  interaction on the OR scale.
- **Small-to-moderate strata counts** — asymptotic chi-square is
  reasonable when each cell has ≥ 5.

## When NOT to use

- **Sparse cells** (0s common) — Breslow-Day is better behaved;
  Zelen's exact for very small n.
- **Continuous exposures** — use logistic regression with
  interaction terms.
- **Only ONE stratum** — no homogeneity concept.

## Assumptions & caveats

- **Independence** — subjects independent within and across strata.
- **Haldane correction** — add 0.5 to all cells if any zero;
  changes point OR mildly, keeps variance defined.
- **Woolf vs Breslow-Day** — Woolf uses adjusted OR variance;
  Breslow-Day is a score-style test with better small-sample
  properties.
- **Non-rejection is not proof of homogeneity** — under-powered
  studies routinely fail to detect real heterogeneity.

## Related in this repo

- `cochran-mantel-haenszel`, `breslow-day` — companion pooling /
  homogeneity tests.
- `chi-square-tests`, `fisher-exact`, `cramers-v-phi` — categorical
  siblings.
- `meta-analysis`, `meta-regression` — larger heterogeneity
  frameworks.
- `logistic-regression` — model-based effect-modification analysis.

## Run

```
python techniques/woolf-homogeneity-of-or/python/woolf_homogeneity_of_or.py
Rscript techniques/woolf-homogeneity-of-or/r/woolf_homogeneity_of_or.R
```

**Refs:** Woolf, B. "On estimating the relationship between blood group and disease." *Annals of Human Genetics*, 19(4): 251-253, 1955; Breslow, N.E. & Day, N.E. *Statistical Methods in Cancer Research, Vol. 1*, IARC, 1980.

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

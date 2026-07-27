# McNemar's Test for Paired Binary Outcomes (Reference §8.2, §8.18)

The paired-samples analogue of the **chi-square / z-test on two proportions**. Each subject provides *two* binary responses (before/after, matched case/control, two raters), giving a 2×2 table **of pairs** rather than of individuals.

```
                     After
              +    |    -
       +----+---------+
Before | +  |  a  |  b
       +----+---------+
       | -  |  c  |  d
       +----+---------+
```

`H₀`: marginal proportions of "+" are equal, i.e. **b = c**. Only the **discordant** pairs (`b`, `c`) carry information; the concordant pairs (`a`, `d`) drop out.

## Statistics computed

| Statistic | Formula | When to use |
|---|---|---|
| **Asymptotic χ²** | `(b − c)² / (b + c)` ~ χ²₁ | `b + c ≥ 25` |
| **Continuity-corrected (Edwards)** | `(|b − c| − 1)² / (b + c)` | `b + c` moderate; conservative |
| **Exact** | two-sided binomial(b + c, 0.5) on `min(b, c)` | small `b + c` |
| **Mid-p** | exact − ½·PMF at the observed count | small samples; better calibrated than plain exact |
| **McNemar OR** | `b / c` (0.5 continuity if zero) | effect size for discordant ratio |
| **Newcombe method 10** | paired 95% CI for `p₁ − p₂` (marginal proportions) | size of the marginal shift |

## Assumptions

- **Paired** binary outcomes; independence *across pairs* (not within).
- Exact/mid-p make no asymptotic assumptions; use them for small `b + c`.

## Files

- `python/mcnemar_test.py` — six statistics above; matches `statsmodels.stats.contingency_tables.mcnemar` exactly for the asymptotic, continuity-corrected, and exact versions.
- `r/mcnemar_test.R` — from-scratch + base `stats::mcnemar.test`.
- `pyspark/mcnemar_test.py` — `groupBy(before, after).count()` aggregation over potentially billions of matched pairs, then a scalar test on the driver. Right pattern when the paired data is too big to `collect()`.

## Run

```
python techniques/mcnemar-test/python/mcnemar_test.py
Rscript techniques/mcnemar-test/r/mcnemar_test.R
python techniques/mcnemar-test/pyspark/mcnemar_test.py
```

**Refs:** McNemar, Q. "Note on the sampling error of the difference between correlated proportions or percentages." *Psychometrika* 12(2), 153–157, 1947; Newcombe, R.G. "Improved confidence intervals for the difference between binomial proportions based on paired data." *Statistics in Medicine* 17(22), 2635–2650, 1998; Agresti, A. *Categorical Data Analysis*, 3rd ed., Wiley, 2013 (Ch. 8).

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

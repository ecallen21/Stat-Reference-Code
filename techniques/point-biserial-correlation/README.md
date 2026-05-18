# Point-Biserial Correlation (Reference §4.4)

Pearson's r between a **continuous** variable `y` and a **truly binary** variable `x` (coded 0/1). Equivalent — algebraically — to:

`r_pb = (ȳ₁ − ȳ₀) / SD(y) · √(p(1 − p))`,

where `ȳ₁`, `ȳ₀` are the group means and `p` = fraction with `x = 1`.

## Same as Student's t-test

`r_pb` and the two-sample Student t-statistic carry the same information:
`t = r_pb · √((n − 2)/(1 − r_pb²))` — identical to the formula on the two groups. So a point-biserial correlation **is** a t-test, just expressed as a correlation coefficient (and as an effect size: `r_pb²` is the variance "explained" by group membership).

Assumes the two groups have similar variance (Student form). For unequal variances, run Welch's t (`techniques/t-tests`) and report Cohen's d (`techniques/effect-sizes`) instead.

## Related — don't confuse them

| Coefficient | x | y | Use |
|-------------|---|---|-----|
| Point-biserial | **truly** binary (0/1) | continuous | This file |
| Biserial | binary that's an artificial dichotomy of a continuous variable (assumes underlying normal) | continuous | Not implemented; see `polycor::biserial` |
| Rank-biserial | binary | ordinal | Mann–Whitney effect size; see `techniques/effect-sizes` |
| Tetrachoric / polychoric | binary / ordinal | binary / ordinal | `techniques/polychoric-correlation` |

## Files
- `python/point_biserial_correlation.py` — from-scratch via the explicit two-group form; compares against `scipy.stats.pointbiserialr` and the equivalent Student t-test.
- `r/point_biserial_correlation.R` — from-scratch + `stats::cor.test` (a Pearson cor.test on `0/1` is the point-biserial).
- PySpark: N/A — for huge data, run the equivalent two-sample t-test on a 0/1 group column (`techniques/t-tests/pyspark/`).

## Run
```
python techniques/point-biserial-correlation/python/point_biserial_correlation.py
Rscript techniques/point-biserial-correlation/r/point_biserial_correlation.R
```

**Ref:** Tate, "Correlation Between a Discrete and a Continuous Variable. Point-Biserial Correlation," *Ann. Math. Stat.* 25, 603–607, 1954.

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

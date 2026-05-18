# Hodges–Lehmann Estimator (Reference §6.29)

Robust nonparametric point estimate of location (or two-sample shift), paired with the rank tests.

| Form | Statistic | Paired with |
|------|-----------|-------------|
| **One-sample** | `median { (xᵢ + xⱼ)/2 : i ≤ j }`  (Walsh averages) | Wilcoxon signed-rank |
| **Two-sample shift** | `median { xᵢ − yⱼ }` | Mann–Whitney U |

The CI for the two-sample shift is read off as **the k-th smallest and k-th largest cross differences**, where `k` is chosen so that the corresponding U falls inside the central `1 − α` region (normal approximation here, exact in scipy / R for small samples).

## Why it's nice

- **Median-of-pairs** is far more efficient than the sample median at the normal (~95.5% asymptotic efficiency vs. the mean) while keeping a 29% breakdown — much higher than the mean's 0%.
- The HL estimate is **what the Wilcoxon / Mann–Whitney "shift" is really estimating** — calling it a "median test" alone misses this point.
- The CI on the shift comes directly from the test you'd already run; no separate bootstrap needed.

## Cost

- One-sample: `O(n²)` Walsh averages.
- Two-sample: `O(n₁ · n₂)` cross differences.

For very large `n`, use the median-of-medians algorithm or sample.

## Files
- `python/hodges_lehmann.py` — from-scratch one- and two-sample HL plus CI via the normal-approximation cutoff.
- `r/hodges_lehmann.R` — from-scratch versions + `stats::wilcox.test(conf.int = TRUE)` (which returns the HL estimate and the inverted-rank-test CI).
- PySpark: N/A — `O(n²)` cross differences are infeasible at Spark scale; sample first.

## Run
```
python techniques/hodges-lehmann/python/hodges_lehmann.py
Rscript techniques/hodges-lehmann/r/hodges_lehmann.R
```

**Ref:** Hodges & Lehmann, "Estimates of Location Based on Rank Tests," *Ann. Math. Stat.* 34(2), 598–611, 1963.

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

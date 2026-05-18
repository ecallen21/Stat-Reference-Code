# Spearman Rank Correlation (Reference §4.2)

Pearson's r computed on the **ranks** of `x` and `y` instead of the raw values. Measures any **monotonic** association — not just linear.

## Formula

When all ranks are distinct (no ties):
`ρ = 1 − 6·Σ dᵢ² / (n(n² − 1))`, where `dᵢ = rank(xᵢ) − rank(yᵢ)`.

With ties, use the general form: Pearson's r on the average ranks (R's `rank()` does this by default). Both Python and R from-scratch implementations here take that route — it works in both cases.

## Why use it over Pearson

- Captures monotone but **nonlinear** relationships that Pearson misses (e.g. `y = exp(x)` — see the demo).
- **Robust to outliers**: ranks bound the influence of any single point.
- Works on **ordinal** data where Pearson's "interval" assumption fails.

The cost: slightly less power than Pearson when the relationship really is linear and the data really are normal.

## Significance and CI

Test `H₀: ρ = 0` via `t = ρ·√((n−2)/(1−ρ²))` on `n − 2` df (same as Pearson, on the ranks). Fisher z CI on ρ: `z = atanh(ρ)`, `SE ≈ 1/√(n−3)`, back-transform with `tanh`. For exact small-sample p-values, scipy and R use a permutation distribution — see `scipy.stats.spearmanr` / `cor.test(method = "spearman", exact = TRUE)`.

## Files
- `python/spearman_rank_correlation.py` — from-scratch average-rank computation, Spearman ρ via Pearson on ranks, t-test, Fisher-z CI; compares against `scipy.stats.spearmanr` and prints Pearson for contrast.
- `r/spearman_rank_correlation.R` — from-scratch + `stats::cor(method = "spearman")` / `stats::cor.test(method = "spearman")`.
- `pyspark/spearman_rank_correlation.py` — rank each column via a window function, then `F.corr` on the rank columns.

## Run
```
python techniques/spearman-rank-correlation/python/spearman_rank_correlation.py
Rscript techniques/spearman-rank-correlation/r/spearman_rank_correlation.R
python techniques/spearman-rank-correlation/pyspark/spearman_rank_correlation.py
```

**Ref:** Spearman, "The Proof and Measurement of Association Between Two Things," *American Journal of Psychology* 15(1), 72–101, 1904.

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

# Mann–Whitney U / Wilcoxon Rank-Sum Test (Reference §6.3)

The nonparametric analogue of the **two-sample t-test**. `H₀`: the two distributions are the same (equivalently, `P(X₁ > X₂) = 0.5`).

## Algorithm

1. Combine both samples; rank the combined values with **average ranks** at ties.
2. `R₁ = sum of ranks in sample 1`.
3. `U₁ = R₁ − n₁(n₁+1)/2`, `U₂ = n₁·n₂ − U₁`.
4. Normal approximation: `E[U₁] = n₁n₂/2`, `Var[U₁] = n₁n₂(n+1)/12 − tie correction`.
5. p-value from `z = (U₁ − E[U₁]) / √Var` with continuity correction.

For small samples scipy / R use an exact null distribution; the from-scratch version here uses the normal approximation throughout (matches `scipy.stats.mannwhitneyu` to 12 decimals for the demo).

## Effect size

`rank-biserial r = 2·U₁ / (n₁n₂) − 1` ∈ `[−1, 1]`. Equals Cliff's δ. Available pre-computed in `techniques/effect-sizes`.

## Assumptions / interpretation

- **Independent samples.**
- The test is sensitive to **any** distributional difference, but is canonically interpreted as a **shift / location** test only when the two groups have **the same shape** under H₀. With different shapes, a significant result means *stochastic dominance* of one group, which can correspond to scale or shape differences, not just shift.
- Continuous data (ties allowed; the tie correction handles them).

## Files
- `python/mann_whitney.py` — from-scratch combined ranks, U computation, normal-approximation p-value with tie correction and continuity correction; reports rank-biserial effect size. Matches `scipy.stats.mannwhitneyu` exactly.
- `r/mann_whitney.R` — from-scratch + base `wilcox.test`.
- `pyspark/mann_whitney.py` — average-rank window function + `groupBy → sum/count`, then closed-form U on the driver. The right pattern for billion-row A/B-style nonparametric tests.

## Run
```
python techniques/mann-whitney/python/mann_whitney.py
Rscript techniques/mann-whitney/r/mann_whitney.R
python techniques/mann-whitney/pyspark/mann_whitney.py
```

**Refs:** Mann & Whitney, "On a Test of Whether One of Two Random Variables Is Stochastically Larger Than the Other," *Ann. Math. Stat.* 18(1), 50–60, 1947; Wilcoxon (1945), same idea via rank sums.

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

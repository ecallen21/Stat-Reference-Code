# Wilcoxon Signed-Rank Test (Reference §6.2)

Test that the distribution of `(X − m₀)` is **symmetric around 0**. For paired data, apply to the differences `X₁ − X₂`. The standard nonparametric replacement for the one-sample / paired t-test.

## Algorithm

1. Compute `dᵢ = xᵢ − m₀`.
2. Drop ties (`dᵢ = 0`).
3. Rank `|dᵢ|` (ties get average ranks).
4. Form `W⁺ = Σ rank(|dᵢ|) over i with dᵢ > 0`. (Signed-rank statistic: `W = Σ sign(dᵢ)·rank(|dᵢ|)`.)
5. Under H₀: `E[W⁺] = n(n+1)/4`, `Var[W⁺] = n(n+1)(2n+1)/24 − tie correction`.
6. Normal-approximation p-value with continuity correction (matches scipy / R's `wilcox.test(..., exact = FALSE)`).

For small `n` and no ties, an exact null distribution is available; scipy and R use it automatically.

## Assumptions

- Differences are **continuous** (a discrete distribution gives ties → reduced power).
- Differences are **symmetric around the null** under H₀. **Drop this and use the sign test** if the differences are clearly skewed.
- Pairs are independent (between, not within).

## Vs. paired t-test

- **t-test** assumes the differences are approximately **normal**.
- **Wilcoxon** assumes **symmetry** (a weaker condition).
- Both are asymptotically Pitman-efficient at the normal; Wilcoxon has higher power for heavy-tailed differences.

## Files
- `python/wilcoxon_signed_rank.py` — from-scratch ranks with average-ties, signed-rank statistic, normal-approximation p with tie-corrected variance and continuity correction; compares against `scipy.stats.wilcoxon`.
- `r/wilcoxon_signed_rank.R` — from-scratch + base `wilcox.test`.
- PySpark: N/A — small-to-medium sample test.

## Run
```
python techniques/wilcoxon-signed-rank/python/wilcoxon_signed_rank.py
Rscript techniques/wilcoxon-signed-rank/r/wilcoxon_signed_rank.R
```

**Ref:** Wilcoxon, "Individual Comparisons by Ranking Methods," *Biometrics Bulletin* 1(6), 80–83, 1945.

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

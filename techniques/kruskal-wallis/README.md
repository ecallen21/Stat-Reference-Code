# Kruskal–Wallis Test (Reference §6.4)

Nonparametric **one-way ANOVA**: H₀ says all `k` groups come from the same distribution.

## Algorithm

1. Combine all groups; rank with average ranks at ties.
2. `Rᵢ` = rank sum for group `i`, with `nᵢ` observations.
3. `H = 12/(N(N+1)) · Σ Rᵢ²/nᵢ  −  3(N+1)`.
4. Tie correction: `H_corr = H / (1 − Σ(t³−t)/(N³−N))`.
5. Under H₀: `H_corr ~ χ²_{k−1}` (asymptotic).

## Vs. one-way ANOVA

- ANOVA: assumes normality + homoscedasticity; tests **mean equality**.
- Kruskal–Wallis: ranks only, no normality, but interpretable as **mean rank** or **stochastic dominance**.
- For continuous data without outliers and roughly normal residuals, ANOVA has slightly more power. With heavy tails or outliers, K–W wins.

## Post-hoc

A significant K–W only says "at least one group differs." For pairwise follow-ups, the nonparametric analogues of Tukey HSD / Dunnett:

- **Dunn's test** (§6.14, deferred).
- **Nemenyi** (§6.30, deferred).
- Pairwise Mann–Whitney with Bonferroni / Holm correction (`techniques/multiple-comparisons`) is a simple, defensible option.

For **ordered** alternatives (group 1 < group 2 < group 3), prefer the **Jonckheere–Terpstra test** (`techniques/jonckheere-terpstra`) — it's more powerful than K–W against a monotone alternative.

## Files
- `python/kruskal_wallis.py` — from-scratch with average-rank ties and full tie correction; matches `scipy.stats.kruskal` exactly.
- `r/kruskal_wallis.R` — from-scratch + base `kruskal.test`.
- PySpark: N/A (small-to-medium-sample test).

## Run
```
python techniques/kruskal-wallis/python/kruskal_wallis.py
Rscript techniques/kruskal-wallis/r/kruskal_wallis.R
```

**Ref:** Kruskal & Wallis, "Use of Ranks in One-Criterion Variance Analysis," *JASA* 47(260), 583–621, 1952.

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

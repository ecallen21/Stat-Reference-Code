# Levene / Brown-Forsythe Variance Test (Reference §47.61)

Levene (1960); Brown & Forsythe (1974). Robust tests for the
equality of variances across k groups. Apply a one-way ANOVA
F-test to the absolute deviations
`Z_{ij} = |Y_{ij} − center_j|`, where `center_j` is the group

    * mean         → Levene
    * median       → Brown-Forsythe (recommended; robust to skew)
    * trimmed mean → Brown-Forsythe variant.

More robust than Bartlett's test or the classical F ratio when data
are non-normal.

## Files

- `python/levene_brown_forsythe.py` — from-scratch W statistic
  and F(k−1, N−k) p-value with all three centering options. Demo
  (n=50 per group):
  - equal-var normals → p = 0.58 (fail to reject)
  - σ = (1, 2, 3.5)   → p < 10⁻⁶ (reject)
  - t₃ same-variance   → p = 0.99 (correct: no false positive
    from heavy tails with median-centered BF)
- `r/levene_brown_forsythe.R` — `car::leveneTest`,
  `onewaytests::bf.test` (R); `scipy.stats.levene`, from-scratch
  (Python).

## When to use

- **Diagnostic before ANOVA / t-test** — variance homogeneity is
  a key assumption.
- **Skewed / heavy-tailed groups** — use median-centered BF.
- **Small samples** — BF has better Type I error than Bartlett.

## When NOT to use

- **Only two groups with normal data** — plain F for variances is
  fine.
- **Very small n per group** — permutation Levene / Fligner-Killeen
  more powerful.
- **Trended variance** (heteroskedasticity in a regression) — use
  Breusch-Pagan / White tests instead.

## Assumptions & caveats

- **Group-level location** — Levene uses mean; heavily skewed
  groups will inflate false positives (BF-median fixes).
- **Independence of observations** — repeated measures need
  variance tests adapted to correlated data.
- **Effect direction** — the test does not say which group is more
  variable; follow with pairwise F ratios / bootstrap CIs.
- **Not a substitute** for using variance-robust inference (Welch's
  t, HC-SE, Satterthwaite df).

## Related in this repo

- `normality-tests`, `outlier-tests`, `equivalence-testing-tost`,
  `moods-median` — assumption-check toolkit.
- `t-tests`, `one-way-anova`, `repeated-measures-anova`,
  `manova` — the mean-comparison tests whose assumption this
  checks.
- `sandwich-robust-se`, `newey-west-hac` — heteroskedasticity-
  robust regression inference.
- `wild-cluster-bootstrap`, `wilcoxon-signed-rank`,
  `mann-whitney` — nonparametric alternatives.

## Run

```
python techniques/levene-brown-forsythe/python/levene_brown_forsythe.py
Rscript techniques/levene-brown-forsythe/r/levene_brown_forsythe.R
```

**Refs:** Levene, H. "Robust tests for equality of variances." In *Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling*, pp. 278-292, 1960; Brown, M.B. & Forsythe, A.B. "Robust tests for the equality of variances." *JASA* 69(346): 364-367, 1974.

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

# Friedman Test (Reference §6.5)

Nonparametric **repeated-measures one-way ANOVA**. H₀: the `k` treatments have the same distribution within each subject (block).

Common use: `n` subjects each measured under `k` conditions; "does treatment matter?"

## Algorithm

1. Rank **within each row** (across treatments), average ranks at ties.
2. `Rⱼ` = sum of column-j ranks across subjects.
3. `F = 12/(nk(k+1)) · ΣⱼRⱼ²  −  3n(k+1)`.
4. Tie correction: divide by `1 − Σ(t³−t)/(n(k³−k))`.
5. Under H₀: `F ~ χ²_{k−1}` (asymptotic).

## Kendall's W

The closely related **coefficient of concordance** is `W = F_corr / (n(k−1))`. It's the natural effect size: 0 = no agreement among subjects' rankings of treatments, 1 = perfect agreement. Calls out separately in §4.6 / `techniques/intraclass-correlation`-adjacent territory.

## When to use

- Repeated measurements on the same subject under different conditions.
- Matched-block designs.
- The data are continuous or ordinal; you don't want to assume normality.

For **two** treatments per subject, the equivalent test is **Wilcoxon signed-rank** (`techniques/wilcoxon-signed-rank`).

## Post-hoc

- Pairwise Wilcoxon signed-rank with Bonferroni / Holm correction (`techniques/multiple-comparisons`).
- **Nemenyi** for all-pairs Friedman post-hoc (deferred — §6.30).

## Files
- `python/friedman_test.py` — from-scratch within-row ranking with average-tie correction; matches `scipy.stats.friedmanchisquare` exactly. Reports Kendall's W.
- `r/friedman_test.R` — from-scratch + base `friedman.test`.
- PySpark: N/A (subjects × treatments matrix is small).

## Run
```
python techniques/friedman-test/python/friedman_test.py
Rscript techniques/friedman-test/r/friedman_test.R
```

**Ref:** Friedman, "The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance," *JASA* 32(200), 675–701, 1937.

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

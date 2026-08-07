# Tukey HSD, Dunnett, Scheffé (Reference §6.9)

Post-hoc multiple-comparison procedures for one-way ANOVA. After the omnibus F-test rejects, follow up with adjusted pairwise comparisons that control the family-wise error rate.

## Tukey HSD (Honest Significant Difference)

All pairwise mean comparisons. Statistic:

```
q = |μ̂_i − μ̂_j| / √(MSE / n_h)      n_h = harmonic mean of (n_i, n_j)
```

Compared against the **studentized range** distribution `q(k, df_within)`. FWER controlled at α across all `k(k−1)/2` pairs.

## Dunnett's test

Each treatment compared to a **single control**. Uses the correlated-t distribution across `k − 1` comparisons — more powerful than Tukey when only control vs each treatment matters.

## Scheffé's method

All possible **linear contrasts** (not just pairwise). Very conservative; the only advantage is joint coverage over the whole contrast space.

## Files

- `python/tukey_hsd.py` — from-scratch Tukey HSD via scipy's `studentized_range` distribution + Sidak-approximated Dunnett. Demo (4 groups, n = 20 each, true means 0.0, 0.5, 1.5, 0.2): Tukey p-values match `scipy.stats.tukey_hsd` to 4 decimals; correctly flags A-C, B-C, C-D as significant.
- `r/tukey_hsd.R` — base `TukeyHSD(aov(...))` + `multcomp::glht(fit, linfct = mcp(g = "Dunnett"))` for the canonical R interface.

## When to use each

- **Tukey HSD** — default for all pairwise comparisons after ANOVA.
- **Dunnett** — comparing each treatment against a control (dose-response, treatment vs placebo).
- **Scheffé** — arbitrary contrasts specified after seeing the data.
- **Bonferroni** / Holm — small number of pre-specified comparisons.

## Assumptions & caveats

- **Equal variances across groups** — Tukey HSD assumes homoscedasticity. Use Tukey-Kramer for unbalanced designs.
- **Normally-distributed residuals**; robust to mild deviations for large `n`.
- **Independence of observations** — for repeated measures, use mixed-model contrasts (`emmeans::pairs`).

## Run

```
python techniques/tukey-hsd/python/tukey_hsd.py
Rscript techniques/tukey-hsd/r/tukey_hsd.R
```

**Refs:** Tukey, J.W. "The problem of multiple comparisons." Unpublished manuscript, Princeton, 1953; Dunnett, C.W. "A multiple comparison procedure for comparing several treatments with a control." *JASA* 50(272), 1096–1121, 1955; Scheffé, H. "A method for judging all contrasts in the analysis of variance." *Biometrika* 40(1/2), 87–110, 1953.

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

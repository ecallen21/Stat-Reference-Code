# Functional ANOVA (Reference §31.4)

Test whether K groups share the **same mean curve** μ_g(t).

## Point-wise F(t)

```
F(t)  =  ( SS_between(t) / (K − 1) )  /  ( SS_within(t) / (n − K) )
```

## Global tests

- **sup-F**  `F_sup = max_t F(t)` — permutation p-value.
- **integrated-F**  `F_int = ∫ F(t) dt` — permutation p-value.
- **L² norm** of pairwise mean differences (Cuevas 2004).

Permutation test: shuffle group labels `B` times; empirical p = fraction
of shuffled `F_sup` (or `F_int`) `≥` observed.

## When to use

- **Groups of curves**: treated vs control growth curves, ECGs by
  condition, sensor time series by device.
- **Any time you want group-comparison FDA** with valid p-values.
- **Distribution-free inference** via permutation.

## When NOT to use

- **Confounders present** — use `functional-ancova` / covariate-
  adjusted FDA.
- **Highly-warped curves** — align first with `curve-registration`.

## Files

- `python/functional_anova.py` — from-scratch pointwise F +
  permutation p (500 permutations). Demo:
  * **DIFFER** — 3 groups with vertical shift ±0.8: `sup F = 160`,
    `p = 0.000`.
  * **EQUAL** — 3 groups from the same mean curve: `sup F = 6.7`,
    `p = 0.17`.
- `r/functional_anova.R` — `fda::Fperm.fd`, `fda.usc::anova.RPm` (R);
  `scikit-fda` (Python).

## Assumptions & caveats

- **Point-wise F is not distributed as F** — the marginal F
  distribution ignores serial correlation across `t`; permutation p
  is the safe statistic.
- **Multiple comparisons across t** — control via Bonferroni or
  cluster-based permutation (Nichols 2003).
- **Sample size per group** — pointwise variance is noisy for small n.
- **Permutation exchangeability** — groups must be exchangeable under
  H0.
- **Extensions**: two-way FANOVA, functional ANCOVA (§31.5), mixed-
  effects FANOVA.

## Related in this repo

- `functional-pca`, `functional-regression`, `functional-clustering`,
  `curve-registration`, `functional-depth` — FDA family (this batch).
- `ancova` — the scalar ANOVA cousin.
- `permutation-test`, `bootstrap` (if present) — the resampling
  toolbox.
- `multiple-comparisons`, `fdr` — the correction machinery.

## Run

```
python techniques/functional-anova/python/functional_anova.py
Rscript techniques/functional-anova/r/functional_anova.R
```

**Refs:** Ramsay, J.O. & Silverman, B.W. *Functional Data Analysis*, Springer, 2005 (Ch. 13); Cuevas, A., Febrero, M. & Fraiman, R. "An ANOVA test for functional data." *Computational Statistics and Data Analysis*, 2004.

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

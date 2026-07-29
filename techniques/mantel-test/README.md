# Mantel Test (Reference §9.18)

Correlation between **two distance matrices** `X` and `Y` on the same `n` subjects — measured in two different ways. Tests whether "close in X-space" implies "close in Y-space".

```
Mantel r = Pearson correlation of upper-triangle entries of X and Y

p-value  = fraction of permuted r (rows/cols of Y shuffled)
           whose |r| >= |observed r|
```

Direct correlation p-values are invalid because entries within a distance matrix are dependent; the permutation test respects that dependence by relabeling **subjects**, not entries.

## Applications

- **Ecology** — genetic distance vs geographic distance across populations.
- **Epidemiology** — dissimilarity in symptom profiles vs biomarker profiles.
- **Consistency** — do two clustering distances agree?
- **Morphometrics** — landmark shape distance vs environmental distance.

## Partial Mantel (Smouse-Long-Sokal 1986)

Correlation between `X` and `Y` **after partialling out** a third matrix `Z`. Regress the upper-triangle entries of `X` on `Z` and of `Y` on `Z`; correlate the residuals; permute for the p-value.

## Files

- `python/mantel_test.py` — from-scratch Mantel and partial Mantel with row/col permutation. Demo: correlated location and feature distances give r = 0.55, p = 0.001.
- `r/mantel_test.R` — wrappers around `vegan::mantel` and `vegan::mantel.partial`.

## Assumptions

- Both matrices are symmetric with zero diagonal and share the same subject ordering.
- Same n in each matrix; entries are dissimilarities (any metric).
- Enough permutations — at least 999 for a reportable p-value; the minimum reachable is `1 / (1 + n_perm)`.

## Run

```
python techniques/mantel-test/python/mantel_test.py
Rscript techniques/mantel-test/r/mantel_test.R
```

**Refs:** Mantel, N. "The detection of disease clustering and a generalized regression approach." *Cancer Res.* 27(2), 209–220, 1967; Smouse, P.E., Long, J.C. & Sokal, R.R. "Multiple regression and correlation extensions of the Mantel test of matrix correspondence." *Syst. Zool.* 35(4), 627–632, 1986.

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

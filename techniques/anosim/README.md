# ANOSIM — Analysis of Similarities (Reference §9.19)

Rank-based non-parametric analog of one-way ANOVA on a **distance matrix**. Clarke (1993). Widely used in ecology, and a common companion to PERMANOVA.

## Statistic

Convert upper-triangle distances to ranks (1 = closest pair). Let

```
r_W = mean rank of pairs in the SAME group
r_B = mean rank of pairs in DIFFERENT groups
R   = (r_B − r_W) / (N / 2)          N = n(n−1)/2 pairs
```

`R ∈ [−1, 1]`:
- `R > 0` — within-group distances shorter than between → groups separate.
- `R ≈ 0` — no group structure in the ranked distances.
- `R < 0` — within-group larger than between (rare; usually indicates mis-labeling).

p-value: permute group labels many times, count `R_perm ≥ R_obs`.

## ANOSIM vs PERMANOVA

| | ANOSIM | PERMANOVA |
|---|---|---|
| Basis | ranks of distances | squared distances |
| Robust to outliers? | yes | less so |
| Uses magnitude of distance? | no | yes |
| Sensitive to dispersion? | yes | yes |
| Effect size | R | pseudo-F, R² |

Report the rank-based test when distances are ordinal or heavy-tailed; report PERMANOVA when magnitudes are meaningful and normality-ish.

## Files

- `python/anosim.py` — from-scratch R statistic with label-permutation p-value; demo shows R = 0.48, p = 0.001 for 3 shifted clusters.
- `r/anosim.R` — thin wrapper around `vegan::anosim`.

## Assumptions

- Symmetric distance matrix, zero diagonal, same subject order in `D` and `groups`.
- Homogeneous dispersions across groups — like PERMANOVA, a "significant" ANOSIM can reflect either location or spread differences. Test dispersions with `vegan::betadisper`.

## Run

```
python techniques/anosim/python/anosim.py
Rscript techniques/anosim/r/anosim.R
```

**Refs:** Clarke, K.R. "Non-parametric multivariate analyses of changes in community structure." *Aust. J. Ecol.* 18(1), 117–143, 1993.

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

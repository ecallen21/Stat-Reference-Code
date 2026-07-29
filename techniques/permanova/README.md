# PERMANOVA — Permutational MANOVA (Reference §9.17)

Non-parametric MANOVA that works on any **distance matrix** — Euclidean, Bray-Curtis, Jaccard, Gower, precomputed dissimilarity. Anderson (2001).

## Pseudo-F statistic

```
SS_total   = Σ_{i < j} d_ij² / n
SS_within  = Σ_g  Σ_{i < j ∈ g} d_ij² / n_g
SS_between = SS_total − SS_within

pseudo-F   = (SS_between / (K − 1)) / (SS_within / (n − K))
```

p-value: permute group labels many times; count how often permuted F ≥ observed F.

## When to prefer over classical MANOVA

- Non-normal / rank-based data.
- Ecological community data (species counts with Bray-Curtis).
- Multivariate outcomes with mixed types.
- Precomputed dissimilarities from a domain-specific metric.

## Files

- `python/permanova.py` — from-scratch pseudo-F + label permutation; recovers p = 0.001 on synthetic 2-group data.
- `r/permanova.R` — thin wrapper around `vegan::adonis2`.

## Assumptions

- **Homogeneity of multivariate dispersions**: significant PERMANOVA can reflect either location or dispersion differences. Test dispersion with `vegan::betadisper` (or the analogous test); if dispersions differ, PERMANOVA's "location" interpretation is confounded.

## Run

```
python techniques/permanova/python/permanova.py
Rscript techniques/permanova/r/permanova.R
```

**Refs:** Anderson, M.J. "A new method for non-parametric multivariate analysis of variance." *Austral Ecol.* 26(1), 32–46, 2001; McArdle, B.H. & Anderson, M.J. "Fitting multivariate models to community data: a comment on distance-based redundancy analysis." *Ecology* 82(1), 290–297, 2001.

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

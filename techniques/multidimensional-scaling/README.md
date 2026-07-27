# Multidimensional Scaling — Classical and Non-Metric (Reference §9.32)

Given a pairwise **distance matrix** `D` between `n` objects (from any dissimilarity — Euclidean, Jaccard, subjective ratings, edit distance, …), MDS finds coordinates in a low-dimensional space (typically 2 or 3) such that the pairwise distances approximate `D` as closely as possible.

## Two variants

### Classical (metric) MDS — Torgerson–Gower

1. Square the distances: `D²`.
2. Double-center: `B = −½ H·D²·H` with `H = I − (1/n) 1 1'`.
3. Eigendecompose `B`; take top-`k` eigenvalues/vectors.
4. Coordinates = eigenvectors · `√eigenvalues`.

When `D` **is** the Euclidean distance matrix of some data, classical MDS recovers that data (up to rotation/reflection) — equivalent to PCA on the raw data.

### Non-metric MDS — Kruskal (1964)

Preserves only the **order** of distances — useful for ordinal/subjective similarity ratings.

Iterate:
1. Given current coords, compute fitted distances `d̂`.
2. Isotonic (monotone) regression of `d̂` on `D` → disparities.
3. Move coords via a Guttman transform to reduce Kruskal's stress-1.
4. Repeat.

```
stress-1 = √( Σ (d̂ − disparity)² / Σ d̂² )
```

Interpretation: `< 0.05` excellent · `0.05–0.10` good · `0.10–0.20` fair · `> 0.20` poor.

## Files

- `python/multidimensional_scaling.py` — classical MDS + non-metric MDS with a from-scratch pool-adjacent-violators isotonic step. Non-metric stress = 0 on Euclidean input (as expected).
- `r/multidimensional_scaling.R` — from-scratch classical + `stats::cmdscale` + `MASS::isoMDS` non-metric.

## Assumptions

- Classical MDS assumes `D` is (approximately) an embeddable Euclidean distance. Non-Euclidean `D` may produce negative eigenvalues — the top positive ones are used but the fit is approximate.
- Non-metric MDS makes no distance assumption beyond ordinal preservation.

## Run

```
python techniques/multidimensional-scaling/python/multidimensional_scaling.py
Rscript techniques/multidimensional-scaling/r/multidimensional_scaling.R
```

**Refs:** Torgerson, W.S. "Multidimensional scaling: I. Theory and method." *Psychometrika* 17(4), 401–419, 1952; Kruskal, J.B. "Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis." *Psychometrika* 29(1), 1–27, 1964; Borg, I. & Groenen, P.J.F. *Modern Multidimensional Scaling*, 2nd ed., Springer, 2005.

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

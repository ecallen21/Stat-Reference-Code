# Correspondence Analysis (Reference §8.5)

The categorical-data analogue of PCA: an SVD-based decomposition of the χ² association in a two-way contingency table that lets you plot row and column categories in a shared low-dimensional space. Rows/columns that co-occur *more than expected under independence* land close together; those that co-occur *less* than expected land far apart.

## The algorithm (compact)

Given an I × J count matrix `N` with grand total `n`:

```
P   = N / n                             (joint probabilities)
r   = row sums,   c = col sums          (row / column masses)
S   = D_r^{-1/2} (P − r c') D_c^{-1/2}   (chi-square-metric residuals)
S   = U Σ V'                             (SVD)

Row principal coords: F = D_r^{-1/2} U Σ
Col principal coords: G = D_c^{-1/2} V Σ
Total inertia = Σ σ_k² = χ² / n
```

Non-trivial dimensions: `min(I, J) − 1`. Each is a slice of χ² that captures a distinct association pattern.

## What you can do with the output

- **Biplot**: plot rows and columns in the first two principal dimensions. Category proximity indicates association.
- **Inertia decomposition**: how much of the total χ² does each dimension explain — a Kaiser-rule scree analog for categorical data.
- **Multiple Correspondence Analysis (MCA)**: `mca_burt()` applies CA to the Burt matrix `Z'Z` of a multi-factor indicator design, extending CA to K > 2 categorical variables.

## Files

- `python/correspondence_analysis.py` — from-scratch SVD-based CA + MCA via the Burt matrix; row/col principal + standard coordinates; total inertia + per-dimension % explained; cross-check against `prince.CA` when the optional `prince` package is installed.
- `r/correspondence_analysis.R` — from-scratch + `MASS::corresp` (or `ca::ca` if installed).

## Assumptions

- Non-negative counts (frequency table); no cells need to be zero, but a zero row or column will be dropped (mass = 0).
- No probability model assumed — CA is descriptive geometry, not an inferential test. Use `chi-square-tests` or `log-linear-models` for significance.

## Run

```
python techniques/correspondence-analysis/python/correspondence_analysis.py
Rscript techniques/correspondence-analysis/r/correspondence_analysis.R
```

**Refs:** Greenacre, M.J. *Correspondence Analysis in Practice*, 3rd ed., Chapman & Hall/CRC, 2017; Benzécri, J.-P. *L'analyse des correspondances*, Dunod, 1973; Nenadic, O. & Greenacre, M. "Correspondence analysis in R, with two- and three-dimensional graphics: the ca package." *J. Stat. Soft.* 20(3), 2007.

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

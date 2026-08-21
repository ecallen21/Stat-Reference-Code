# Spatial Weights Matrix (Reference §23.2)

The `(n × n)` matrix `W` encodes which locations are "neighbours". `W_ij` = weight of location `j` when computing statistics at `i`. `W_ii = 0`.

## Common constructions

- **Contiguity** — `W_ij = 1` if regions share an edge (rook) or edge/vertex (queen). Polygon-based.
- **Distance band** — `W_ij = 1` if `d_ij ≤ threshold`.
- **k-Nearest Neighbours** — `W_ij = 1` if `j` is among `i`'s `k` closest points.
- **Kernel-weighted** — `W_ij = exp(−d² / (2h²))` (Gaussian) or `(1 − u²)²` for `u = d/h < 1` (bisquare).

## Row standardization

```
W_ij ← W_ij / Σ_k W_ik
```

Each row sums to 1. Standard for Moran's I, spatial regression models.

## Files

- `python/spatial_weights_matrix.py` — distance-band, kNN, and Gaussian / bisquare kernel constructors with optional row-standardization. Demo (20 random points): distance-band with threshold 2 averages 1.70 neighbours per location; kNN with k=4 gives exactly 4; Gaussian kernel rows sum to 1.
- `r/spatial_weights_matrix.R` — pointers to `spdep::poly2nb`, `spdep::knn2nb`, `spdep::dnearneigh`, `spdep::nb2listw`.

## Choosing W

- **Polygon data** (counties, tracts) → contiguity (queen is default).
- **Point data** → kNN or distance band.
- **Continuous kernel weights** → GWR-style analysis; also for kernel-based Moran variants.
- **Row-standardize** for interpretability (row sum = 1 means the "average neighbour value").

## Assumptions & caveats

- **W is a modelling choice** — sensitivity analysis across k or bandwidth is essential.
- **Symmetric?** — Rook / Queen / distance-band `W` are symmetric before row-standardization; kNN is generally asymmetric even before.
- **Islands** — locations with no neighbours (isolated island, tiny threshold) get zero rows; handle explicitly.

## Run

```
python techniques/spatial-weights-matrix/python/spatial_weights_matrix.py
Rscript techniques/spatial-weights-matrix/r/spatial_weights_matrix.R
```

**Refs:** Cliff, A.D. & Ord, J.K. *Spatial Autocorrelation*, Pion, 1973; Anselin, L. *Spatial Econometrics: Methods and Models*, Kluwer, 1988.

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

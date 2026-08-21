# Local Moran's I / LISA (Reference §23.4)

Anselin 1995 — decomposes the global Moran's I into per-location contributions:

```
I_i = z_i · Σ_j W_ij · z_j            z = (x − x̄) / sd(x)
```

Sum of `I_i` over `i` gives the global I (up to a normalizing constant).

## Cluster / outlier categorization

Compare each `x_i` vs mean and its neighbours' weighted mean:

| Category  | Own value | Neighbours | I_i sign | Interpretation      |
|-----------|-----------|------------|----------|---------------------|
| **HH**    | high      | high       | +        | hot-spot cluster    |
| **LL**    | low       | low        | +        | cold-spot cluster   |
| **HL**    | high      | low        | −        | spatial outlier     |
| **LH**    | low       | high       | −        | spatial outlier     |

## Files

- `python/local_moran_lisa.py` — local Moran's I per location + cluster/outlier type + permutation p-values (permuting each location's neighbours in turn). Demo on 8×8 grid with a planted 3×3 hot spot (9 cells): correctly identifies 8 significant HH locations.
- `r/local_moran_lisa.R` — `spdep::localmoran`, `rgeoda::local_moran`.

## When to use

- **Cluster mapping** — visualize where hot / cold spots occur, not just whether they exist globally.
- **Spatial outlier detection** — HL / LH points that deviate from their surroundings.
- **Companion to global Moran's I** — always report LISA alongside global for a complete picture.

## Assumptions & caveats

- **Multiple testing** — every location is a hypothesis. Apply FDR (`multiple-testing-corrections`) if many locations.
- **Sensitivity to W** — different weight matrices give different cluster maps.
- **Sample-size dependence** — permutation p-values need enough neighbours; kNN with k ≥ 6 recommended.

## Related

- **Getis-Ord G_i^*** — alternative local statistic; positive/negative interpretation differs.
- **SATScan** / spatial-scan (`spatial-scan-cluster`) — cluster-detection via a scan window.
- **Bayesian smoothing** — priors on rates for reliable local-rate estimation.

## Run

```
python techniques/local-moran-lisa/python/local_moran_lisa.py
Rscript techniques/local-moran-lisa/r/local_moran_lisa.R
```

**Refs:** Anselin, L. "Local indicators of spatial association—LISA." *Geogr. Anal.* 27(2), 93–115, 1995; Getis, A. & Ord, J.K. "The analysis of spatial association by use of distance statistics." *Geogr. Anal.* 24(3), 189–206, 1992.

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

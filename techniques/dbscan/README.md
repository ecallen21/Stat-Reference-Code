# DBSCAN — Density-Based Spatial Clustering (Reference §9.11)

**Ester, Kriegel, Sander & Xu (1996).** Finds clusters as regions of high point density separated by regions of low density. Unlike k-means, it doesn't require you to specify the number of clusters, handles arbitrarily shaped clusters, and explicitly labels **outliers** as noise.

## Two parameters

- **`eps`**: radius of a point's neighborhood.
- **`min_pts`**: minimum number of points (including the point itself) inside `eps` for the point to be a **core** point.

## Three point types

- **Core**: `≥ min_pts` neighbors within `eps` — sits inside a dense region.
- **Border**: has fewer than `min_pts` neighbors but lies within `eps` of some core point.
- **Noise**: neither. Not assigned to any cluster; labeled `−1`.

## Algorithm

For each unvisited point:
1. Compute its `eps`-neighborhood.
2. If not core, mark as noise (may later be reclassified as border).
3. Otherwise, start a new cluster; expand it via BFS through connected core points and pull in their border neighbors.

## Choosing `eps`

The standard heuristic is the **k-distance graph**: for each point, compute the distance to its k-th nearest neighbor; sort ascending; plot. The value at the visual "knee" is a reasonable `eps` for `min_pts = k + 1`. `k_distance_graph()` in the Python file computes the sorted list.

## Files

- `python/dbscan.py` — from-scratch BFS-expansion DBSCAN + k-distance-graph helper. Cluster counts and noise counts match `sklearn.cluster.DBSCAN` exactly.
- `r/dbscan.R` — from-scratch + `dbscan::dbscan` when installed.

## Assumptions

- Density is meaningful in the ambient space — standardize features on very different scales first.
- `eps` and `min_pts` must be chosen; the k-distance graph is a data-driven starting point but the values still need substantive judgment.
- Density-based method — **struggles** with clusters of *very different densities* (a single global `eps` can't cover both). For that, use HDBSCAN (extension; not implemented here).

## Run

```
python techniques/dbscan/python/dbscan.py
Rscript techniques/dbscan/r/dbscan.R
```

**Refs:** Ester, M., Kriegel, H.-P., Sander, J. & Xu, X. "A density-based algorithm for discovering clusters in large spatial databases with noise." *KDD '96*, 226–231, 1996; Schubert, E., Sander, J., Ester, M., Kriegel, H.-P. & Xu, X. "DBSCAN revisited, revisited: why and how you should (still) use DBSCAN." *ACM TODS* 42(3), 1–21, 2017.

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

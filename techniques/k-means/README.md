# k-means Clustering (Reference §9.9)

Partitions `n` observations into `k` clusters by iteratively:

1. **Assign** each point to its nearest current centroid.
2. **Update** each centroid to the mean of its assigned points.

Repeat until assignments stop changing. This is **Lloyd's algorithm**. The objective it minimizes is the **within-cluster sum of squares** (WCSS / inertia):

```
W  =  Σ_c  Σ_{x ∈ cluster c}  ‖x − μ_c‖²
```

Convergence is to a **local** minimum — initialization matters.

## Initialization

- **Random** — pick k points at random. Fast but noisy; may hit poor local minima.
- **k-means++** — first centroid random; each subsequent one picked with probability proportional to `D(x)²` (squared distance to the nearest picked centroid). Provably ~O(log k) worse than optimal on average; the default in scikit-learn.
- **k-means||** (used by MLlib) — the parallel/distributed analog of k-means++.

Standard practice: run several restarts with k-means++ and keep the lowest-inertia solution.

## Choosing k

Not part of the fit. See `cluster-validation` for silhouette, gap statistic, Calinski–Harabasz, and Davies–Bouldin — the standard k-selection criteria.

## Files

- `python/k_means.py` — Lloyd's algorithm from scratch + k-means++ init + multi-restart. Inertia matches `sklearn.cluster.KMeans` to 12 dp on the demo.
- `r/k_means.R` — thin wrapper around base `stats::kmeans` (Lloyd or Hartigan-Wong).
- `pyspark/k_means.py` — MLlib `KMeans` (uses k-means|| init) + silhouette from `ClusteringEvaluator`.

## Assumptions

- **Spherical, equal-variance clusters** — that's what SS minimization implicitly assumes. Non-spherical / uneven-variance clusters need GMM (see `gaussian-mixture-models`) or DBSCAN.
- **Euclidean distance** — k-means only makes sense with a squared-Euclidean objective. For general distances, use k-medoids (PAM).
- Standardize features before fitting if they're on different scales.

## Run

```
python techniques/k-means/python/k_means.py
Rscript techniques/k-means/r/k_means.R
python techniques/k-means/pyspark/k_means.py
```

**Refs:** Lloyd, S.P. "Least squares quantization in PCM." *IEEE Trans. Inf. Theory* 28(2), 129–137, 1982; MacQueen, J. "Some methods for classification and analysis of multivariate observations." *5th Berkeley Symp. Math. Stat. Prob.*, 1967; Arthur, D. & Vassilvitskii, S. "k-means++: The advantages of careful seeding." *SODA*, 2007.

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

# Gap Statistic for Choosing K (Reference §47.81)

Tibshirani, Walther & Hastie (2001). Compare observed within-
cluster dispersion `W_k` to a null reference from a uniform
distribution over the data's bounding box:

    Gap(k) = E*[log W_k*] − log W_k
    se_k   = sd(log W_k*) · √(1 + 1/B).

Pick the SMALLEST k with `Gap(k) ≥ Gap(k+1) − se_{k+1}`. Handles
K=1 (no clusters) cleanly, unlike elbow / silhouette.

## Files

- `python/gap_statistic_cluster.py` — from-scratch Gap
  computation with within-dispersion + Tibshirani-Walther-Hastie
  1-SE rule. Demo:
  - 3-cluster data: Gap jumps −0.08 → +2.53 at k=3 → **selected K=3**.
  - Uniform data: 1-SE rule → **selected K=1** (no structure).
- `r/gap_statistic_cluster.R` — `cluster::clusGap`,
  `factoextra::fviz_gap_stat` (R); `gap-statistic`, from-scratch
  (Python).

## When to use

- **Choosing K** in k-means / k-medoids / GMM / hierarchical.
- **Ruling out K=1** — most other criteria don't handle it.
- **When you can afford B ≥ 20 bootstraps of reference data**.
- **Complement Hopkins statistic** — first "any structure?" then "how many?".

## When NOT to use

- **Very large n** — expensive; use k-means-approximate versions.
- **Bounding-box mis-specification** for non-rectangular support —
  sample from PCA-uniform-on-loading or convex-hull instead.
- **Highly imbalanced clusters** — Gap can overestimate K; combine
  with silhouette / DB.

## Assumptions & caveats

- **Null model** — uniform over PCA-rotated bounding box is Tibshirani's
  suggestion; a plain box (used here) is simpler but conservative.
- **Bootstrap variance** — larger B reduces se; B = 20–50 usual.
- **Log-dispersion scale** — use natural log; avoid divide-by-zero for
  singleton clusters.
- **1-SE rule** favours parsimony; use "max Gap" for a more aggressive K.

## Related in this repo

- `silhouette-clusters`, `davies-bouldin-index`,
  `hopkins-clusterability` — companion cluster diagnostics.
- `k-means`, `k-medoids`, `gaussian-mixture-models`,
  `hierarchical-clustering`, `spectral-clustering`,
  `hdbscan-clustering`, `affinity-propagation` — clustering algorithms
  Gap can select.
- `bootstrap-optimism-correction`, `nonparametric-bootstrap` —
  bootstrap machinery underpinning the reference distribution.
- `information-criteria` — BIC / AIC alternatives.

## Run

```
python techniques/gap-statistic-cluster/python/gap_statistic_cluster.py
Rscript techniques/gap-statistic-cluster/r/gap_statistic_cluster.R
```

**Refs:** Tibshirani, R., Walther, G. & Hastie, T. "Estimating the number of clusters in a data set via the gap statistic." *JRSS-B* 63(2): 411-423, 2001.

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

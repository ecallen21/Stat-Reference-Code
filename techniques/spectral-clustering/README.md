# Spectral Clustering (Reference §47.78)

Ng, Jordan & Weiss (2001). Build a symmetric-normalised Laplacian
from an affinity graph, take the K smallest eigenvectors as a
non-linear embedding, and k-means the row-normalised rows to
recover labels. Captures non-convex cluster shapes (moons, rings)
that fail k-means directly.

## Files

- `python/spectral_clustering.py` — from-scratch spectral
  clustering with local-scale bandwidth heuristic + k-NN sparsification
  + row-normalised k-means on eigenvector rows. Demo:
  - Two moons (n=600): k-means ARI 0.32, **spectral ARI 1.000**
  - Concentric rings (n=600): k-means ARI ≈0, **spectral ARI 1.000**.
- `r/spectral_clustering.R` — `kernlab::specc`, `RSpectra` (R);
  `sklearn.cluster.SpectralClustering`, from-scratch (Python).

## When to use

- **Non-convex clusters** — manifolds, rings, moons.
- **Graph / network partitioning** — the normalised-cut objective.
- **Image segmentation** — Shi-Malik (1997) foundational.
- **Community detection** at moderate n.

## When NOT to use

- **Very large n** — O(n²) affinity matrix + eigen-decomposition;
  use Nyström / power iteration variants.
- **High-dim data with heavy noise** — construct kNN affinity or
  denoise first.
- **Streaming data** — spectral clustering assumes a static
  distance matrix.

## Assumptions & caveats

- **Bandwidth σ** critical — median-heuristic often too large for
  tight clusters; local-scale (Zelnik-Manor & Perona 2005) is
  robust.
- **Number of eigenvectors K** = target number of clusters; eigengap
  heuristic can help pick K.
- **Row normalisation** in Ng-Jordan-Weiss is important for the
  final k-means step.
- **Symmetric-normalised vs random-walk Laplacian** — both work;
  Shi-Malik uses the random-walk form.

## Related in this repo

- `k-means`, `k-medoids`, `hierarchical-clustering`,
  `gaussian-mixture-models`, `dbscan`, `hdbscan-clustering`,
  `affinity-propagation` — clustering peers.
- `hopkins-clusterability`, `silhouette-clusters`,
  `davies-bouldin-index`, `gap-statistic-cluster` — validation.
- `isomap`, `lle-locally-linear-embedding`, `diffusion-maps`,
  `tsne-umap` — manifold / graph-based dimension-reduction cousins.
- `stochastic-block-model`, `latent-space-network` — network
  clustering alternatives.

## Run

```
python techniques/spectral-clustering/python/spectral_clustering.py
Rscript techniques/spectral-clustering/r/spectral_clustering.R
```

**Refs:** Ng, A.Y., Jordan, M.I. & Weiss, Y. "On spectral clustering: Analysis and an algorithm." *NeurIPS*, 2001; Von Luxburg, U. "A tutorial on spectral clustering." *Statistics and Computing* 17(4): 395-416, 2007.

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

# t-SNE + UMAP (Reference §26.5)

Nonlinear dimensionality reduction for **visualization** of high-dimensional data in 2 or 3 dimensions. Both preserve local neighborhood structure — clusters stay clusters — but distort global distances.

## t-SNE (van der Maaten & Hinton 2008)

```
1. Pairwise probabilities p_ij in high-D via Gaussians tuned to perplexity (~5–50).
2. Pairwise probabilities q_ij in low-D via a Student-t kernel (heavy tails).
3. Gradient-descent minimize KL(P ‖ Q).
```

**Perplexity** is the effective number of neighbors per point; default 30, sensitive.

## UMAP (McInnes & Healy 2018)

Based on Riemannian geometry / topological data analysis. Faster than t-SNE, tends to preserve **global** structure better, deterministic by default. Key parameters:

- **`n_neighbors`** (~15) — local vs global emphasis.
- **`min_dist`** (~0.1) — how tightly points cluster.

## Files

- `python/tsne_umap.py` — wraps `sklearn.manifold.TSNE` and `umap-learn` (optional) + a cluster-purity diagnostic on the embedding. Demo (3 well-separated 20-D Gaussians): t-SNE gives cluster purity 1.0.
- `r/tsne_umap.R` — `Rtsne::Rtsne` and `uwot::umap` (canonical R implementations).

## When to use

- **Exploratory visualization** of high-D data (embeddings, images, single-cell RNA-seq).
- **Cluster discovery** — as a starting point, follow with DBSCAN or K-means on the embedding.
- **Model diagnostics** — visualize learned representations of a neural network.

## When NOT to use

- **As a distance-preserving dimension reduction** — use PCA / MDS / Isomap.
- **For downstream statistical inference** on the embedded coordinates — they are algorithmically constructed, not meaningful features.
- **Reversible dimension reduction** — no explicit projection, no inverse transform.

## Cautions

- **Cluster sizes and inter-cluster distances are not meaningful** — the algorithms distort them.
- **Local minima** — run multiple seeds; report a representative one.
- **Perplexity / n_neighbors sensitive** — try several values before drawing conclusions.
- **Do not run k-means / density-based clustering on the embedding** for anything beyond visualization; cluster in the original space.

## Run

```
python techniques/tsne-umap/python/tsne_umap.py
Rscript techniques/tsne-umap/r/tsne_umap.R
```

**Refs:** van der Maaten, L.J.P. & Hinton, G. "Visualizing data using t-SNE." *JMLR* 9, 2579–2605, 2008; McInnes, L., Healy, J. & Melville, J. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." arXiv:1802.03426, 2018.

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

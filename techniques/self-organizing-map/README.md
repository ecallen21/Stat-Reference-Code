# Self-Organizing Map / Kohonen Network (Reference §47.311)

Kohonen (1982). Unsupervised 2-D grid of neurons that learns
a TOPOLOGICALLY ORDERED representation of the input:

```
for each x:
    BMU = argmin_i ‖x − w_i‖              (best matching unit)
    for each neuron j:
        w_j ← w_j + η · h(j, BMU) · (x − w_j)
```

Neighbourhood `h` decays with distance from BMU (Gaussian
kernel) and with training epoch. Widely used for visualisation
(U-matrix) and exploratory clustering.

## Files

- `python/self_organizing_map.py` — 6×6 grid on 3 well-
  separated 4-D clusters. After 30 epochs, cluster labels
  populate contiguous grid regions and the U-matrix shows
  boundary values (~1.8-2.9) between clusters vs
  intra-cluster (~0.3-0.5). Classic Kohonen visual.
- `r/self_organizing_map.R` — `kohonen`, `som` (R); `minisom`,
  `susi`, from-scratch (Python).

## When to use

- **Exploratory clustering** with 2-D visualisation.
- **Topological mapping** for interpretable structure.
- **Data-quality checks** — mis-labelled points fall in the
  "wrong" grid cell.

## When NOT to use

- **When accurate clustering is needed** — k-means / HDBSCAN
  usually beat SOM.
- **Very high-dimensional inputs** — curse of dimensionality
  hits the distance metric.
- **Real-time updates required** — batch SOM is preferred but
  requires re-training.

## Assumptions & caveats

- **Grid size** — √N ≈ 5-10× fewer than samples.
- **Neighbourhood decay** — Gaussian h(t) = exp(−d²/(2σ(t)²));
  σ shrinks over epochs.
- **Learning rate** — decays over epochs; too fast → poor
  organisation.
- **U-matrix interpretation** — high values indicate cluster
  boundaries.

## Related in this repo

- `k-means`, `k-medoids`, `hierarchical-clustering`,
  `dbscan`, `hdbscan-clustering`, `spectral-clustering` —
  clustering alternatives.
- `tsne-umap`, `isomap`, `lle-locally-linear-embedding`,
  `diffusion-maps` — dimensionality-reduction visualisations.
- `hopfield-network` — related associative-memory NN.

## Run

```
python techniques/self-organizing-map/python/self_organizing_map.py
Rscript techniques/self-organizing-map/r/self_organizing_map.R
```

**Refs:** Kohonen, T. "Self-organized formation of topologically correct feature maps." *Biological Cybernetics*, 43(1): 59-69, 1982; Kohonen, T. *Self-Organizing Maps*, 3rd ed., Springer, 2001.

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

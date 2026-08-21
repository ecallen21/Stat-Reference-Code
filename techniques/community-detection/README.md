# Community Detection (Reference §24.3)

Partition graph nodes into densely-connected groups. **Modularity** (Newman-Girvan) scores a partition against a degree-preserving random null:

```
Q = (1 / 2m) · Σ_ij [A_ij − k_i k_j / (2m)] · 1{c(i) = c(j)}
```

Positive Q → more within-community edges than chance.

## Algorithms implemented

- **Greedy agglomerative modularity** (Clauset-Newman-Moore): start with singletons; repeatedly merge the pair whose merge maximises ΔQ.
- **Spectral 2-way split** (Newman): sign of the leading eigenvector of the modularity matrix `B_ij = A_ij − k_i k_j / (2m)`.

The library-strength approaches (**Louvain**, **Leiden**) are similar but move nodes between communities rather than merging clusters wholesale. Use `igraph::cluster_leiden` for real work — Louvain has a well-known degenerate-community problem that Leiden fixes.

## Related methods

| Approach | Idea |
|---|---|
| **Louvain** | modularity + local move + community aggregation |
| **Leiden** | Louvain + refinement pass; guarantees connected communities |
| **Walktrap** | random walks tend to stay inside communities |
| **Infomap** | minimum description length of a random walker's path |
| **Stochastic block model** | generative fit (see `stochastic-block-model`) |
| **Spectral clustering** | k-means on Laplacian eigenvectors (see `graph-embedding-spectral`) |

## When to use

- **Exploratory** — first pass on any medium-sized network.
- **Choosing k** — modularity-based methods don't need `k` upfront.
- **Comparing partitions** — normalised mutual information (NMI), adjusted Rand.

## Files

- `python/community_detection.py` — from-scratch modularity, greedy agglomerative + spectral 2-way, NetworkX cross-check. Demo (3-block SBM, within-p 0.8, between-p 0.05, 10+10+10 nodes): greedy Q = 0.5722, recovers all 3 communities at 100% clustering accuracy; NetworkX greedy matches Q exactly.
- `r/community_detection.R` — `igraph::cluster_louvain / cluster_leiden / cluster_fast_greedy / cluster_walktrap / cluster_infomap`.

## Assumptions & caveats

- **Resolution limit** — modularity cannot detect communities smaller than a scale that depends on `m`; use resolution parameter (`γ ≠ 1`) or multi-scale variants.
- **Greedy is O(n² · m)** — expensive above a few thousand nodes; use Louvain / Leiden for scale.
- **Q is monotone in merging trivial pairs** — the algorithm can produce many small clusters when the graph has hubs; report per-cluster size.
- **Different runs** may return different partitions with near-identical Q — near-degenerate landscape.
- **Ground truth vs Q** — high modularity doesn't guarantee recovery of a planted structure; validate with NMI on labelled test data.

## Run

```
python techniques/community-detection/python/community_detection.py
Rscript techniques/community-detection/r/community_detection.R
```

**Refs:** Newman, M.E.J. & Girvan, M. "Finding and evaluating community structure in networks." *Phys. Rev. E* 69, 026113, 2004; Clauset, A., Newman, M.E.J. & Moore, C. "Finding community structure in very large networks." *Phys. Rev. E* 70, 066111, 2004; Traag, V.A., Waltman, L. & van Eck, N.J. "From Louvain to Leiden: guaranteeing well-connected communities." *Sci. Rep.* 9, 5233, 2019.

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

# Spectral Graph Embedding (Reference §24.9)

Represent each node as a low-dimensional vector by taking eigenvectors of a
graph matrix. Three classical routes:

| Embedding | Matrix | Vectors used |
|---|---|---|
| **Laplacian eigenmaps** | `L = D − A` | bottom d non-trivial eigenvectors |
| **Normalized Laplacian** | `L_sym = I − D^{-1/2} A D^{-1/2}` | bottom d non-trivial eigenvectors |
| **Adjacency spectral** (ASE) | `A` | top d eigenvectors, scaled by `√|λ|` |

Downstream uses:

- **k-means on the embedding** → **spectral clustering** (equivalent to relaxed graph-cut / RatioCut / normalized cut).
- **Inner-product decoder** → link prediction (`P̂(edge) ≈ σ(⟨u_i, u_j⟩)`).
- **Node features** for a downstream classifier.
- **Visualisation** (`d = 2`).

## Non-spectral alternatives

Modern practice adds **learned** embeddings:

- **DeepWalk / node2vec** — Skip-Gram on random-walk sequences.
- **LINE** — first- and second-order proximity via edge sampling.
- **Graph neural networks** — GCN, GraphSAGE, GAT, GIN aggregate features across neighbourhoods; supervised or self-supervised.
- **UMAP / t-SNE on Laplacian embedding** — combines with `graph-embedding-spectral` when a purely-visual 2-D layout is needed.

## When to use

- **Small / medium graphs** where a full eigendecomposition is feasible (`n < 10⁴`).
- **Interpretable analysis** — spectral gaps directly diagnose community structure.
- **Warm start** for iterative methods (SBM, k-means, GNNs).
- **Bipartite / signed / directed graphs** — use appropriate matrix (biadjacency, signed Laplacian, transition).

## Files

- `python/graph_embedding_spectral.py` — from-scratch Laplacian (normalized + unnormalized) and adjacency spectral embeddings + k-means; scikit-learn `SpectralEmbedding` cross-check. Demo (3-block SBM n=45, within-p 0.5, between-p 0.05, d=2): normalized Laplacian 100% cluster accuracy; unnormalized 100%; adjacency spectral 93%; sklearn 100%.
- `r/graph_embedding_spectral.R` — `igraph::embed_laplacian_matrix / embed_adjacency_matrix`, `kernlab::specc`.

## Assumptions & caveats

- **Spectral gap** — clustering quality depends on the gap between eigenvalues `λ_d` and `λ_{d+1}`. Small gap → unstable embedding.
- **Disconnected graphs** produce multiple zero Laplacian eigenvalues — one per component; extract the largest component first or work per-component.
- **Choice of Laplacian** — normalized (Ng-Jordan-Weiss) is the default; unnormalized biases toward balanced clusters via **RatioCut**.
- **Sign of eigenvectors is arbitrary** — align signs across runs if reproducibility matters.
- **Scalability** — eigendecomp is `O(n³)` dense; use Lanczos (`scipy.sparse.linalg.eigsh`, `LOBPCG`) for large sparse graphs.

## Run

```
python techniques/graph-embedding-spectral/python/graph_embedding_spectral.py
Rscript techniques/graph-embedding-spectral/r/graph_embedding_spectral.R
```

**Refs:** Belkin, M. & Niyogi, P. "Laplacian eigenmaps for dimensionality reduction and data representation." *Neural Comput.* 15, 1373–1396, 2003; Ng, A.Y., Jordan, M.I. & Weiss, Y. "On spectral clustering: analysis and an algorithm." *NIPS*, 2001; von Luxburg, U. "A tutorial on spectral clustering." *Stat. Comput.* 17, 395–416, 2007.

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

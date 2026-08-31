# Locally Linear Embedding (Reference §25.15)

Roweis & Saul (2000). Preserves **local geometry**: each point is
reconstructed as a linear combination of its k-nearest neighbours, and
the same weights should reconstruct it in the low-dim embedding.

## Algorithm

```
1. k-NN for each point.
2. W = argmin Σ_i || x_i − Σ_{j∈N(i)} W_ij x_j ||²    s.t. Σ_j W_ij = 1.
3. Y = argmin tr( Yᵀ (I − W)ᵀ (I − W) Y )              s.t. YᵀY = I.
   Take the bottom d + 1 eigenvectors of M = (I − W)ᵀ (I − W); drop
   the trivial (all-ones) eigenvector.
```

## When to use

- **Preserve local neighbourhoods** — cluster boundaries, tightly-
  connected substructure.
- **Alternative to Isomap** — LLE avoids the global-shortcut problem
  when short-cut edges are noisy.
- **Fast + spectral** — no gradient descent.

## When NOT to use

- **Global geometry matters** — LLE distorts long-range distances;
  Isomap or MDS better.
- **Isolated clusters** — LLE can collapse them.
- **High noise** — sensitive to `k` and the neighbourhood graph.

## Files

- `python/lle_locally_linear_embedding.py` — from-scratch k-NN,
  local-reconstruction weights, sparse eigendecomposition. Demo on a
  3-D S-curve (n = 200, k = 12): recovered axes correlate |0.62| with
  roll parameter and |0.89| with height. LLE is known to be more
  distorted globally than Isomap on this manifold.
- `r/lle_locally_linear_embedding.R` — `lle`, `RDRToolbox`, `dimRed`
  (R); `sklearn.manifold.LocallyLinearEmbedding` (Python).

## Assumptions & caveats

- **Choice of `k`** — small `k` = poor global structure; large `k` =
  averaging kills local geometry.
- **Regularisation** — the local Gram matrix `C` is singular when
  `k > d` in ambient space; add a small `reg · tr(C) · I / k`
  (Roweis-Saul).
- **Sign of embedding axes** — arbitrary.
- **Modified LLE** (Zhang & Wang 2007) improves stability for `k > d`;
  **Hessian LLE** (Donoho & Grimes 2003) preserves the intrinsic
  geometry better on curved manifolds.
- **Compare with Isomap** — LLE is faster but less accurate globally.

## Related in this repo

- `isomap` — global-distance-preserving sibling.
- `tsne-umap` — modern non-spectral manifold-learning.
- `kernel-pca` — spectral non-linear DR.
- `variational-autoencoder`, `contrastive-learning` — deep learning
  alternatives.

## Run

```
python techniques/lle-locally-linear-embedding/python/lle_locally_linear_embedding.py
Rscript techniques/lle-locally-linear-embedding/r/lle_locally_linear_embedding.R
```

**Refs:** Roweis, S.T. & Saul, L.K. "Nonlinear dimensionality reduction by locally linear embedding." *Science*, 2000; Donoho, D.L. & Grimes, C. "Hessian eigenmaps: locally linear embedding techniques for high-dimensional data." *PNAS*, 2003.

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

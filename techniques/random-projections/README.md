# Random Projections (Reference §25.12)

**Johnson-Lindenstrauss (1984)**: projecting `n` points from `ℝ^d` to
`ℝ^k` via a random matrix preserves all pairwise distances within
factor `(1 ± ε)` with high probability, for `k = O(log n / ε²)`.

## Formula

```
Y = X R,       R_{ij} iid Gaussian  N(0, 1/k)  or  Achlioptas {−√3, 0, +√3}/√k.
```

Data-independent, streaming-friendly, distance-preserving.

## When to use

- **Massively high-dim data** (text, images, genomics) where PCA is
  too expensive.
- **Streaming / online** — no data pass needed.
- **Similarity-based downstream tasks** (k-NN, LSH, clustering) — the
  JL guarantee protects distances directly.

## When NOT to use

- **You need interpretable directions** — R is random, not
  meaningful.
- **Very small n** — JL bound is loose; PCA gives better distortion
  for the same `k`.

## Files

- `python/random_projections.py` — Gaussian + Achlioptas projections;
  distortion sweep (mean, min, max ratios) across `k ∈ {20, 50, 100,
  200}`; mean distortion drops from 12 % to 4 %. Cluster-label 1-NN
  accuracy: 1.00 in both original `d = 500` and projected `k = 50`.
- `r/random_projections.R` — `RandPro` (R); `sklearn.random_projection`
  (Python).

## Assumptions & caveats

- **k choice** — the JL bound is `k ≥ 8 ln(n) / ε²` for `1 ± ε`
  distortion.
- **Distortion is worst-case** — typical distortions are far smaller.
- **Sparse variants** (Li 2006) reduce compute to `O(k · nnz(X))`.
- **Combine with hashing** for locality-sensitive-hashing pipelines.

## Related in this repo

- `dimensionality-reduction-pca`, `probabilistic-pca`, `sparse-pca`,
  `kernel-pca` — data-dependent alternatives.
- `isomap`, `lle-locally-linear-embedding`, `diffusion-maps`,
  `tsne-umap` — manifold-preserving DR.
- `variational-autoencoder`, `autoencoder`, `contrastive-learning` —
  neural DR.

## Run

```
python techniques/random-projections/python/random_projections.py
Rscript techniques/random-projections/r/random_projections.R
```

**Refs:** Johnson, W.B. & Lindenstrauss, J. "Extensions of Lipschitz mappings into a Hilbert space." *Contemporary Mathematics*, 1984; Achlioptas, D. "Database-friendly random projections." *PODS*, 2001; Li, P., Hastie, T. & Church, K. "Very sparse random projections." *KDD*, 2006.

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

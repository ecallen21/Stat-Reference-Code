# Isomap (Reference §25.14)

Tenenbaum, de Silva & Langford (2000). Preserves **geodesic distances**
(shortest paths along a k-NN graph) rather than ambient Euclidean
distance. First manifold-learning method to scale to real high-dim data.

## Algorithm

1. Build a **k-NN graph** on `X` with Euclidean weights.
2. Compute all-pairs shortest paths `D_geo` (Dijkstra / Floyd-Warshall).
3. Apply **classical MDS** to `D_geo`: double-centre `−½ D_geo²`,
   take top-`d` eigenvectors × `√eigenvalue`.

## When to use

- **Nonlinear manifolds embedded in high-dim space** — Swiss roll,
  hand-written digit strokes, face image collections.
- **Global structure matters** — Isomap preserves long-range distances
  better than LLE / local kernel methods.
- **Downstream visualisation / clustering** in the embedded space.

## When NOT to use

- **Sparse / noisy graph** — short-cuts destroy geodesic structure.
- **Multiple disconnected components** — geodesic distance is infinite.
- **Very large n** — `O(n² log n)` shortest-path is expensive; use
  L-Isomap or Landmark Isomap.

## Files

- `python/isomap.py` — from-scratch k-NN graph + Floyd-Warshall +
  classical MDS. Demo on the **Swiss roll** (n = 200, k = 10):
  first Isomap coord correlates **|0.99|** with roll parameter,
  second **|0.80|** with height. PCA on the same data would tangle
  the two intrinsic coordinates.
- `r/isomap.R` — `RDRToolbox`, `dimRed`, `vegan::isomap` (R);
  `sklearn.manifold.Isomap`, `megaman` (Python).

## Assumptions & caveats

- **Choice of `k`** — small `k` = disconnected graph; large `k` =
  short-cuts through the ambient space.
- **Sensitivity to noise** — a single noisy edge can dramatically
  shorten geodesic distances (short-cut problem).
- **Isometry assumption** — data must lie on an isometric manifold; a
  crumpled sheet works, but a stretched manifold does not.
- **Out-of-sample extension** requires triangulation (Bengio 2004).
- **Landmark Isomap** — pick `L ≪ n` landmarks; O(nL log n).

## Related in this repo

- `tsne-umap` — sibling manifold-learning family.
- `lle-locally-linear-embedding` — local-structure-preserving alternative.
- `kernel-pca` — kernel-based non-linear DR.
- `variational-autoencoder`, `contrastive-learning` — deep alternatives.

## Run

```
python techniques/isomap/python/isomap.py
Rscript techniques/isomap/r/isomap.R
```

**Refs:** Tenenbaum, J.B., de Silva, V. & Langford, J.C. "A global geometric framework for nonlinear dimensionality reduction." *Science*, 2000; de Silva, V. & Tenenbaum, J.B. "Global versus local methods in nonlinear dimensionality reduction." *NeurIPS*, 2003.

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

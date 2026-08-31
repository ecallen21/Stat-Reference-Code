# Diffusion Maps (Reference §25.16)

Coifman & Lafon (2006). Nonlinear DR that preserves **diffusion
distances** — `t`-step random-walk distances on a Gaussian similarity
graph.

## Algorithm

```
1. Build K_ij = exp(−‖x_i − x_j‖² / (2 σ²)).
2. Row-normalise:  P = D⁻¹ K   (Markov transition matrix).
3. Symmetrise via M = D^{-1/2} K D^{-1/2}, eigendecompose.
4. Embedding = top-k non-trivial eigenvectors × λ_k^t.
```

`t = 0` gives spectral embedding; larger `t` emphasises long-range
structure.

## When to use

- **Manifold learning with noise** — integrates over all paths, not
  just shortest.
- **Multi-scale structure** — `t` controls the scale.
- **Single-cell RNA-seq pseudotime** — `destiny` / `scanpy.tl.diffmap`.

## When NOT to use

- **Simple linear structure** — PCA is enough.
- **Very large n** — full eigendecomposition is `O(n³)`; use Nyström
  approximation.
- **Extrapolation to new points** — needs Nyström extension.

## Files

- `python/diffusion_maps.py` — from-scratch symmetric-normalisation
  eigendecomposition. Demo: 3-D S-curve manifold (n=200), σ = median
  distance, `t = 1`. Best diffusion coord for `t`: `|corr| = 0.84`;
  for `h`: `|corr| = 0.92` — intrinsic coordinates recovered (up to
  axis permutation).
- `r/diffusion_maps.R` — `diffusionMap`, `RDRToolbox`, `destiny` (R);
  `pyDiffMap`, `datafold`, `scanpy` (Python).

## Assumptions & caveats

- **σ** — set to median pairwise distance; too small = disconnected
  graph.
- **Axis permutation / sign** — like all spectral methods.
- **Trivial eigenvector** — the all-ones eigenvector (`λ = 1`) is
  always skipped.
- **`t` parameter** — larger `t` compresses high-frequency components,
  focuses on cluster structure.
- **Anisotropic variants** — `α`-normalisation (Coifman) separates
  geometry from density.

## Related in this repo

- `isomap`, `lle-locally-linear-embedding`, `tsne-umap` — manifold-
  learning siblings.
- `kernel-pca`, `spectral-clustering` (if present) — spectral cousins.
- `variational-autoencoder`, `autoencoder` — deep alternatives.

## Run

```
python techniques/diffusion-maps/python/diffusion_maps.py
Rscript techniques/diffusion-maps/r/diffusion_maps.R
```

**Refs:** Coifman, R.R. & Lafon, S. "Diffusion maps." *Applied and Computational Harmonic Analysis*, 2006; Nadler, B. et al. "Diffusion maps, spectral clustering and reaction coordinates of dynamical systems." *ACHA*, 2006.

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

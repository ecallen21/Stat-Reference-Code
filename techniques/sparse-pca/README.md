# Sparse PCA (Reference §25.8)

Zou, Hastie & Tibshirani (2006). Classical PCA loadings involve every
variable — a nightmare for interpretation. **Sparse PCA** adds an L1
penalty to drive irrelevant loadings to zero.

## Formulation

View PCA as regression of `X` on itself:

```
min_{A, B}  ‖ X − X B Aᵀ ‖_F²  +  λ Σ_j ‖b_j‖₁      s.t.  AᵀA = I.
```

- Alternating: fix `A` → elastic-net regression column-wise for `B`.
- Fix `B` → refresh `A` via reduced SVD of `Xᵀ X B`.

## When to use

- **Interpretable components** — biology, finance, marketing.
- **High-dim `p ≫ n`** — sparsity regularises.
- **Downstream selection** — sparse loadings act as feature selection.

## When NOT to use

- **Explanatory variance is the only goal** — plain PCA is optimal.
- **All variables are truly informative** — sparsity hurts.
- **Very correlated blocks** — group-sparse PCA extension (Jenatton
  2010) is better.

## Files

- `python/sparse_pca.py` — from-scratch alternating soft-threshold +
  SVD. Demo on synthetic 10-d data with two block-supported latent
  factors (vars 0-2 and 7-9): **plain PCA has 12/20 non-zero loadings;
  sparse PCA drives to exactly 6/20 — the true block support**.
- `r/sparse_pca.R` — `elasticnet::spca`, `nsprcomp`, `PMA::SPC` (R);
  `sklearn.decomposition.SparsePCA` (Python).

## Assumptions & caveats

- **λ tunes sparsity vs variance** — cross-validate on out-of-sample
  reconstruction, or pick by inspection.
- **Orthogonality of `A`** — the algorithm keeps `A` orthonormal;
  loadings `B` may lose orthogonality.
- **Multiple restarts** — non-convex; the solver depends on init.
- **Signed loadings** — the sign of each component is arbitrary; use
  `flip_sign_by_max` if you need canonical signs.
- **Non-negative variant** (NNSPCA, Sigg-Buhmann 2008) enforces
  non-negativity for parts-based interpretation.

## Related in this repo

- `ridge-lasso-elasticnet` — the penalty family used inside.
- `nmf`, `dictionary-learning`, `ica`, `kernel-pca`,
  `variational-autoencoder` — sibling factorisations.
- `high-dimensional-*` (Batch 38) — same sparsity toolbox applied to
  regression.
- `principal-component-analysis` (if present) — the dense parent.

## Run

```
python techniques/sparse-pca/python/sparse_pca.py
Rscript techniques/sparse-pca/r/sparse_pca.R
```

**Refs:** Zou, H., Hastie, T. & Tibshirani, R. "Sparse principal component analysis." *Journal of Computational and Graphical Statistics*, 2006; Witten, D.M., Tibshirani, R. & Hastie, T. "A penalized matrix decomposition, with applications to sparse principal components and canonical correlation analysis." *Biostatistics*, 2009.

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

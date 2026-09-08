# Tensor Decomposition -- Tucker + CP (Reference §6.17)

Tucker (1966); Harshman (1970, CANDECOMP / PARAFAC). Extend SVD to
higher-order tensors `X ∈ ℝ^{I × J × K}`.

## CP (canonical polyadic)

    X ≈ Σ_{r=1}^R  a_r ⊗ b_r ⊗ c_r        (rank R)

Parameters: `R · (I + J + K)`.

## Tucker (higher-order SVD)

    X ≈ G ×_1 U⁽¹⁾ ×_2 U⁽²⁾ ×_3 U⁽³⁾

with core `G ∈ ℝ^{R₁ × R₂ × R₃}` and factor matrices.

## Files

- `python/tensor_decomposition_tucker_cp.py` — CP-ALS
  (Khatri-Rao + LSQ update) + HOSVD from scratch. Demo (8 × 6 × 5
  rank-2 tensor + noise): CP rank 2 relative error 0.21, rank 4
  0.17, Tucker (3, 3, 3) 0.19.
- `r/tensor_decomposition_tucker_cp.R` — `rTensor::cp/tucker/hosvd`,
  `multiway::parafac` (R); `tensorly.decomposition`,
  `scikit-tensor` (Python).

## When to use

- **Multi-way data** — chemometrics (samples × wavelengths ×
  timepoints), recommender systems (users × items × context),
  neuroimaging (subject × voxel × timepoint).
- **Low-rank structure across modes** — reveals shared latent
  factors.
- **Compression** — Tucker core is much smaller than the original
  tensor.

## When NOT to use

- **Two-way data** — use plain SVD / PCA.
- **Non-negative or count data** — use NN-CP / Poisson tensor
  factorisation.
- **Ill-posed CP with high rank** — CP is NP-hard; use non-
  negativity, orthogonality, or Tucker as a stabiliser.

## Assumptions & caveats

- **Rank selection** — for CP, CORCONDIA diagnostic + cross-
  validation; for Tucker, mode-wise SVD singular-value drop.
- **CP uniqueness** — Kruskal-uniqueness condition; not always
  identifiable.
- **Non-convex** — ALS can converge to local optima; multi-start
  restarts.
- **Missing entries** — use weighted / imputation-based ALS.

## Related in this repo

- `pca`, `sparse-pca`, `probabilistic-pca`, `nmf`, `ica` — matrix
  decomposition family.
- `functional-pca` — one-way functional analogue.
- `multivariate-multiple-regression` — matrix-response cousin.

## Run

```
python techniques/tensor-decomposition-tucker-cp/python/tensor_decomposition_tucker_cp.py
Rscript techniques/tensor-decomposition-tucker-cp/r/tensor_decomposition_tucker_cp.R
```

**Refs:** Tucker, L.R. "Some mathematical notes on three-mode factor analysis." *Psychometrika*, 31(3): 279-311, 1966; Harshman, R.A. "Foundations of the PARAFAC procedure." *UCLA Working Papers in Phonetics*, 1970; Kolda, T.G. & Bader, B.W. "Tensor decompositions and applications." *SIAM Review*, 51(3): 455-500, 2009.

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

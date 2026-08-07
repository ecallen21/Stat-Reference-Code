# Kernel PCA (Reference §9.10)

Nonlinear dimensionality reduction (Schölkopf-Smola-Müller 1998). Implicitly lift the data to a high-dimensional feature space via a kernel `k(x, y) = ⟨φ(x), φ(y)⟩`, then run PCA there — without ever materializing `φ(x)`.

## Algorithm

```
1. K_ij = k(x_i, x_j)                        (n × n Gram matrix)
2. K_c  = K − 1_n K − K 1_n + 1_n K 1_n      (center in feature space)
3. K_c v_k = λ_k v_k                          (eigen-decompose)
4. Z_ik = √λ_k · v_k[i]                       (k-th component of x_i)
```

## Common kernels

- **RBF (Gaussian)**: `exp(−γ ‖x − y‖²)` — default. `γ = 1 / (2 σ²)`.
- **Polynomial**: `(xᵀy + c)^d`.
- **Sigmoid**: `tanh(a xᵀy + b)`.

## Files

- `python/kernel_pca.py` — from-scratch RBF and polynomial kernels + centered Gram + eigendecomposition. Demo on two concentric rings (impossible for linear PCA): kernel PCA separates the classes by 1.85 SD on PC1; linear PCA separates by 0.015 SD. Matches `sklearn.decomposition.KernelPCA` exactly.
- `r/kernel_pca.R` — `kernlab::kpca` (production).

## When to use

- **Nonlinear manifold structure** — data lies on a curved surface, not a hyperplane.
- **Preprocessing for downstream classification / clustering** on nonlinearly separable data.
- **Visualization** in 2-D or 3-D.

## Related methods

- **PCA**: linear; the special case `k(x, y) = xᵀy`.
- **t-SNE / UMAP**: focus on local neighborhoods for visualization; different objective than kernel PCA.
- **Isomap / LLE**: manifold-learning; preserve geodesic / local distances.
- **Diffusion maps / spectral clustering**: closely related eigenmap methods.

## Assumptions & caveats

- **Kernel choice** and **γ** matter a lot; cross-validate on downstream performance if possible.
- **Cost**: `O(n²)` memory for the Gram matrix, `O(n³)` for the eigendecomposition. Use **Nyström approximation** for large `n`.
- **Out-of-sample projection**: compute the kernel between the new point and training points, then project as in Step 4 above.
- **Interpretation**: kernel-PCA components don't map to linear combinations of original features; use for downstream tasks, not for interpretability.

## Run

```
python techniques/kernel-pca/python/kernel_pca.py
Rscript techniques/kernel-pca/r/kernel_pca.R
```

**Refs:** Schölkopf, B., Smola, A. & Müller, K.-R. "Nonlinear component analysis as a kernel eigenvalue problem." *Neural Comput.* 10(5), 1299–1319, 1998; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch 14.5.4).

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

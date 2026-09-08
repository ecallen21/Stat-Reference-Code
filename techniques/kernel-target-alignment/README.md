# Kernel Target Alignment (Reference §47.136)

Cristianini, Kandola, Elisseeff & Shawe-Taylor (2001). Model-
agnostic kernel selection metric:

    A(K, K*) = ⟨K_c, K*_c⟩_F / (‖K_c‖_F · ‖K*_c‖_F)

with `K_c` the CENTERED kernel matrix and `K* = y yᵀ` the ideal
kernel from labels. Higher = better; drives RBF bandwidth
selection, multiple-kernel learning weights, and neural-tangent-
kernel diagnostics.

## Files

- `python/kernel_target_alignment.py` — from-scratch centered-
  kernel alignment. Demo (200 pts in 2-D, y = sign(x₀)):
  - σ = 0.05 → A = 0.09
  - σ = 0.30 → A = 0.26
  - **σ = 1-3 → A = 0.47-0.49** (broad peak)
  - Linear kernel → A = 0.45 (nearly as good — this target IS
    linear).
- `r/kernel_target_alignment.R` — `kernlab::kernelPol`,
  `mklaren` (R); `sklearn`, `scikit-multiflow` custom (Python).

## When to use

- **RBF bandwidth selection** without cross-validation.
- **Multiple-kernel learning** weights (sum of kernels).
- **Kernel design / diagnostic** — quantify how "aligned" a
  kernel is to a supervised task.
- **NTK analysis** — measure task-alignment of an infinitely-wide
  net's tangent kernel.

## When NOT to use

- **Regression targets** — extend to centered kernel alignment on
  y yᵀ or use HSIC.
- **Very large n** — O(n²) memory for K; Nyström or random-
  feature approximations.
- **When CV is affordable** and calibrated held-out error preferred.

## Assumptions & caveats

- **Centering** essential — uncentered alignment biased toward
  constant kernels.
- **Binary classification** convention y ∈ {−1, +1}; multi-class
  extensions replace `y yᵀ` with a class-indicator Gram.
- **A ∈ [0, 1]** for positive-semidefinite K.
- **Alignment ≠ accuracy** — high A is necessary but not
  sufficient for good downstream error.

## Related in this repo

- `random-fourier-features`, `nadaraya-watson-kernel-regression`,
  `gaussian-process-regression`, `sparse-gaussian-process`,
  `kernel-pca`, `kernel-density-estimation`,
  `svm-classifier`, `support-vector-regression` — kernel
  methods this feeds into.
- `hsic-independence`, `mmd-two-sample-test`,
  `mutual-information` — kernel-based dependence / two-sample
  measures.
- `information-geometry`, `information-bottleneck` — related
  representation-quality metrics.

## Run

```
python techniques/kernel-target-alignment/python/kernel_target_alignment.py
Rscript techniques/kernel-target-alignment/r/kernel_target_alignment.R
```

**Refs:** Cristianini, N., Kandola, J., Elisseeff, A. & Shawe-Taylor, J. "On kernel-target alignment." *NeurIPS*, 2001; Cortes, C., Mohri, M. & Rostamizadeh, A. "Algorithms for learning kernels based on centered alignment." *JMLR* 13: 795-828, 2012.

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

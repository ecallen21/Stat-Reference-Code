# Random Fourier Features (Reference §47.86)

Rahimi & Recht (2007). Bochner's theorem: any shift-invariant
kernel `k(x, y) = k(x − y)` is a Fourier transform of a probability
measure. Draw D frequencies `w_j ~ p(w)` and phases
`b_j ~ Unif(0, 2π)`; the feature map

    φ(x) = √(2/D) [ cos(w_1'x + b_1), …, cos(w_D'x + b_D) ]

satisfies `E[φ(x)'φ(y)] = k(x, y)`. Any linear model on φ(X)
approximates kernel ridge / SVM at O(nD) memory instead of O(n²).

## Files

- `python/random_fourier_features.py` — RFF for the Gaussian
  RBF kernel + kernel-ridge demo. Demo (n=500, d=8):
  - kernel Frobenius error: D=50 → 0.29; D=200 → 0.14; D=800 → 0.09.
  - kernel-ridge MSE: exact 0.393; RFF D=100 → 0.94; D=500 → 0.45;
    D=2000 → 0.42.
- `r/random_fourier_features.R` — no first-class R package
  besides custom / `kernlab` for base kernels;
  `sklearn.kernel_approximation.RBFSampler / Nystroem` (Python).

## When to use

- **Kernel methods at scale** — n too large for O(n²) memory /
  time.
- **Streaming / online learning** — RFF gives fixed-size vector
  representation.
- **Fast approximate kernel-SVM / kernel-ridge / kernel-PCA**.
- **Any linear-in-features solver** — logistic regression, SVM,
  Lasso — on top of RFF.

## When NOT to use

- **Very small n** — exact kernel is cheap and better.
- **Non-shift-invariant kernels** (polynomial, string, graph) —
  need dedicated methods (Nyström, TensorSketch).
- **When Nyström is available** — often more accurate at the same
  D on data-dependent sampling.

## Assumptions & caveats

- **Bandwidth σ** must be well-chosen — median heuristic on
  pairwise distances is standard.
- **D–accuracy trade-off** — error decays as O(1/√D).
- **Random-projection variance** — average over multiple RFF draws
  for stability.
- **QMC-RFF / structured Fourier** (Yu et al 2016, Le et al 2013)
  reduce variance further.

## Related in this repo

- `kernel-density-estimation`, `kernel-pca`,
  `gaussian-process-regression`, `nadaraya-watson-kernel-regression`,
  `support-vector-regression`, `svm-classifier`, `mmd-two-sample-test`,
  `hsic-independence` — kernel methods RFF speeds up.
- `random-projections`, `product-quantization-pq`,
  `hyperloglog-cardinality` — sketching / hashing family.
- `pca`, `sparse-pca`, `probabilistic-pca` — dimension-reduction
  neighbours.
- `neural-tangent-kernel` — RFF-related theoretical link to NNs.

## Run

```
python techniques/random-fourier-features/python/random_fourier_features.py
Rscript techniques/random-fourier-features/r/random_fourier_features.R
```

**Refs:** Rahimi, A. & Recht, B. "Random features for large-scale kernel machines." *NeurIPS*, 2007; Le, Q., Sarlós, T. & Smola, A. "Fastfood: Approximating kernel expansions in loglinear time." *ICML*, 2013.

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

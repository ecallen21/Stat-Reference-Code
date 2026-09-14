# Nystrom Kernel Approximation (Reference §47.342)

Williams & Seeger (2001). Approximate the `n × n` kernel
matrix `K` by sampling `m ≪ n` landmark points and using the
Nystrom formula:

```
K ~ C @ W⁺ @ Cᵀ
```

where `C = K[:, S]` (n × m) and `W = K[S, S]` (m × m). This
reduces cost from `O(n²)` to `O(nm)` and turns downstream
kernel ridge, kernel PCA, or spectral clustering into a
low-rank problem solvable in `O(nm²)`.

## Files

- `python/nystrom_approximation.py` — RBF kernel on n=800,
  d=5. Full 5.1 MB kernel matrix; Nystrom with m=50 reaches
  2% relative Frobenius error at 0.34 MB, m=200 gets to
  0.2% at 1.6 MB — a ~ 3× memory saving per 10× accuracy.
- `r/nystrom_approximation.R` — `kernlab`, `bigKRLS` (R);
  `sklearn.kernel_approximation.Nystroem`, from-scratch
  (Python).

## When to use

- **Large-n kernel methods** — kernel ridge, kernel PCA,
  kernel k-means when n ≥ 10 000.
- **Spectral clustering** at scale.
- **Feature maps for linear learners** — Nystrom features
  feed a linear SVM or logistic regression.

## When NOT to use

- **Small n** — full kernel is fine.
- **Poor landmarks** — random sampling can fail on manifold
  data; use k-means landmarks or leverage-score sampling.
- **When exact spectrum matters** — approximation error
  scales with excluded eigenvalues.

## Assumptions & caveats

- **Landmark selection** — uniform is a safe baseline; K-means
  landmarks (Zhang-Kwok 2010) or leverage-score sampling
  give provable error bounds.
- **Pinv regularisation** — add `ε I` before inverting `W`
  when near-singular.
- **m scaling** — theoretical guarantees typically need
  `m = Ω(√n log n)` for spectral tasks.
- **Fixed-basis vs adaptive** — random Fourier features
  (Rahimi-Recht) is data-independent; Nystrom is data-
  dependent and usually stronger.

## Related in this repo

- `random-fourier-features` — data-independent alternative.
- `randomized-svd` — general low-rank approximation.
- `kernel-pca`, `gaussian-process-regression`,
  `sparse-gaussian-process` — kernel methods that benefit.
- `spectral-clustering` — direct application.

## Run

```
python techniques/nystrom-approximation/python/nystrom_approximation.py
Rscript techniques/nystrom-approximation/r/nystrom_approximation.R
```

**Refs:** Williams, C.K.I. and Seeger, M. "Using the Nystrom method to speed up kernel machines." In *NeurIPS*, 2001.

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

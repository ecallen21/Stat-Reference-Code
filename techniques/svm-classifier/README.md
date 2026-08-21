# Support Vector Machine (Reference §26.9)

Binary classifier (Cortes & Vapnik 1995) that finds the **maximum-margin hyperplane** between two classes:

```
minimize  ½ ‖w‖² + C Σ_i ξ_i
s.t.       y_i (wᵀ x_i + b) ≥ 1 − ξ_i,   ξ_i ≥ 0
```

`C` trades margin width vs training errors.

## Kernel trick

Replace `xᵀ x'` with `k(x, x')` to fit nonlinear boundaries without explicit feature mapping:

- **Linear**: `xᵀ x'`
- **Polynomial**: `(γ xᵀ x' + r)^d`
- **RBF (Gaussian)**: `exp(−γ ‖x − x'‖²)` — most common default

## Files

- `python/svm_classifier.py` — from-scratch linear SVM via **Pegasos** (Shalev-Shwartz 2007 stochastic subgradient); RBF/polynomial via sklearn (LIBSVM). Demo: linearly separable data → 100% accuracy; concentric-rings RBF SVM → 100%.
- `r/svm_classifier.R` — `e1071::svm` (LIBSVM wrapper) or `kernlab::ksvm`.

## When to use

- **Small-to-moderate n** (`< 10⁴`) with moderate `p` — SVM scales as `O(n²)` in kernel mode.
- **High-dimensional but few samples** — SVM's margin regularization handles `p > n` gracefully.
- **Nonlinear boundaries** — RBF kernel is a strong default before trying trees.

## When NOT to use

- **Very large n** — cubic training cost. Use linear SVM (`LinearSVC`), stochastic-gradient linear classifiers, or trees.
- **Calibrated probabilities matter** — SVM outputs are signed distances, not probabilities. Add Platt scaling or isotonic calibration (see `calibration-scaling`).
- **Interpretability required** — kernel-SVM decision surfaces are opaque.

## Hyperparameters

- **C** — 0.01 to 100 on a log grid; smaller → wider margin, more regularization.
- **γ (RBF)** — inverse length scale; `1 / (n_features · Var(X))` is the sklearn default. Tune jointly with C.

## Assumptions & caveats

- **Standardize features** before SVM — the kernel is scale-sensitive.
- **Class imbalance** — set `class_weight = "balanced"` or subsample (see `class-imbalance`).
- **Multiclass** — one-vs-one (LIBSVM default) or one-vs-rest.

## Run

```
python techniques/svm-classifier/python/svm_classifier.py
Rscript techniques/svm-classifier/r/svm_classifier.R
```

**Refs:** Cortes, C. & Vapnik, V. "Support-vector networks." *Mach. Learn.* 20(3), 273–297, 1995; Shalev-Shwartz, S., Singer, Y., Srebro, N. & Cotter, A. "Pegasos: primal estimated sub-gradient solver for SVM." *Math. Prog.* 127(1), 3–30, 2011.

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

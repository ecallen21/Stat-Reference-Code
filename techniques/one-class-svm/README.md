# One-Class SVM (Reference §21.x extra)

Anomaly / novelty detection where only "normal" data is available at training
time. Schölkopf et al. (2001) find a maximum-margin hyperplane that separates
the training data from the origin in an RBF-kernel feature space:

```
min_{w, ρ, ξ}  ½ ‖w‖² + 1 / (ν n) · Σ ξ_i − ρ
s.t.           ⟨w, φ(x_i)⟩ ≥ ρ − ξ_i,   ξ_i ≥ 0
```

The parameter `ν ∈ (0, 1)` is:

- an **upper bound** on the fraction of training points that lie outside the boundary,
- a **lower bound** on the fraction of support vectors.

Decision: `f(x) = sign(⟨w, φ(x)⟩ − ρ)` — `+1` inlier, `−1` outlier.

## When to use

- **Only normals labelled** — fraud, intrusion, novelty in a stream.
- **Non-Gaussian, non-convex** inlier region — the RBF kernel bends the boundary.
- **Small to medium data** — the QP scales as `O(n²)` – `O(n³)`; use nearest-neighbour / isolation forest for large-scale problems.

## Files

- `python/one_class_svm.py` — KDE-baseline anomaly detector + `sklearn.svm.OneClassSVM` (RBF, `ν=0.05`) cross-check. Demo (400 N(0, I) train + 50 in-distribution + 50 U(−6, 6) outliers): KDE catches 90% outliers at 0% inlier FPR; sklearn OC-SVM catches 90% outliers at 8% inlier FPR; support-vector fraction 0.095 ≈ ν = 0.05.
- `r/one_class_svm.R` — `e1071::svm(type='one-classification')`, `kernlab::ksvm(type='one-svc')`.

## Related methods

- **Isolation forest** — random-tree ensemble; scales better on high-dim data (see `isolation-forest-anomaly`).
- **Local outlier factor (LOF)** — density-ratio to k-nearest-neighbours (`dbscan::lof`).
- **Elliptic envelope** — parametric Gaussian assumption.
- **Robust covariance / MCD** — inlier region defined by Mahalanobis distance.

## Assumptions & caveats

- **ν is a hyperparameter, not the outlier rate** — it caps *training-set* margin violations, not test-time outlier prevalence.
- **Kernel and γ dominate** — cross-validate on a small labelled validation set if you have one; otherwise use `gamma='scale'` and iterate on ν.
- **Feature scaling** matters — always standardize continuous features first.
- **High-dimensional inputs** — RBF kernels concentrate; consider PCA or a deep-feature preprocessor.
- **Streaming / concept drift** — retrain periodically or use online variants; static OC-SVM assumes stationary distribution.

## Run

```
python techniques/one-class-svm/python/one_class_svm.py
Rscript techniques/one-class-svm/r/one_class_svm.R
```

**Refs:** Schölkopf, B. et al. "Estimating the support of a high-dimensional distribution." *Neural Comput.* 13(7), 1443–1471, 2001; Tax, D.M.J. & Duin, R.P.W. "Support vector data description." *Machine Learning* 54(1), 45–66, 2004.

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

# Support Vector Regression (SVR) (Reference §5.17)

Drucker et al. (1996); Smola & Schölkopf (2004). Regression analogue
of SVM with **ε-insensitive** loss:

    L_ε(y, f(x)) = max(0, |y − f(x)| − ε)

and regularisation `‖f‖² / 2`. The dual introduces support vectors
and a kernel `k(x, x')`:

    f(x) = Σᵢ (αᵢ − αᵢ*) k(xᵢ, x) + b

Only points with `|y − f(x)| ≥ ε` have nonzero α.

## Files

- `python/support_vector_regression.py` — projected-gradient
  ε-insensitive dual for RBF kernel from scratch. Demo (n=100,
  `y = sin(x) + 0.2 noise`): in-sample MSE 0.035; predictions
  track `sin(x)` on a grid.
- `r/support_vector_regression.R` — `e1071::svm(type='eps-
  regression')`, `kernlab::ksvm`, `liquidSVM` (R);
  `sklearn.svm.SVR / LinearSVR` (Python).

## When to use

- **Small-to-medium tabular regression** with nonlinear structure.
- **Robustness to noise inside the ε-band** — samples inside are
  not fitted → sparser solution.
- **Kernelised nonlinearity** — RBF, polynomial, or custom kernels
  for exotic domains.

## When NOT to use

- **Very large n** — dual is `O(n²)` memory / `O(n³)` compute;
  use LinearSVR or ensembles.
- **Interpretation needed** — SVR gives a kernel-space predictor;
  no coefficients.
- **Uncertainty quantification** — no built-in CI; use quantile-
  loss variants or bootstrap.

## Assumptions & caveats

- **Feature scaling** — RBF kernel is scale-sensitive; standardise.
- **ε, C, kernel bandwidth** hyperparameters — cross-validate.
- **Sparse dual** — solutions with many active SVs indicate the
  ε-band is too narrow.
- **Not translation-invariant to y** — subtract mean before
  fitting.

## Related in this repo

- `svm-classifier`, `kernel-density-estimation`,
  `nadaraya-watson-kernel-regression`,
  `gaussian-process-regression` — kernel cousins.
- `ridge-lasso-elasticnet`, `nonlinear-regression`,
  `gradient-boosting`, `random-forest` — regression alternatives.

## Run

```
python techniques/support-vector-regression/python/support_vector_regression.py
Rscript techniques/support-vector-regression/r/support_vector_regression.R
```

**Refs:** Drucker, H. et al. "Support vector regression machines." *NIPS*, 1996; Smola, A.J. & Schölkopf, B. "A tutorial on support vector regression." *Statistics and Computing*, 14(3): 199-222, 2004.

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

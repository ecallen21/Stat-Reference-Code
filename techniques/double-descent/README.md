# Double Descent (Reference §47.162)

Belkin, Hsu, Ma & Mandal (2019); Nakkiran et al. (2020). Test error
follows a **double-descent** curve as a function of model / data /
training complexity:

- **Classic regime** (under-parametrised): U-shape bias-variance.
- **Interpolation threshold**: error PEAKS as p ~ n.
- **Modern regime** (over-parametrised): error monotonically drops
  as p ≫ n, often below the classical minimum.

Explained by the minimum-norm solution's implicit bias in
over-parametrised regression.

## Files

- `python/double_descent.py` — min-norm least squares sweep across
  p ∈ {2, …, 500} with n_train = 40, d_true = 10, σ_noise = 0.5:
  - Dip at p = 10 (test MSE ≈ **0.32**).
  - **Peak at p = n = 40** (test MSE = **19.58**, 60× worse).
  - Second descent to test MSE ≈ 4 as p → 500.
- `r/double_descent.R` — pure-R min-norm LS sweep with the same
  setup.

## When to use

- **Understanding modern over-parametrised training** — the
  classical "smaller model, less variance" rule breaks down.
- **Model selection with fixed compute** — going bigger past the
  interpolation threshold can be safer than tuning at the peak.
- **Explaining why huge neural nets generalise**.

## When NOT to use

- **Standard cross-validation** already handles model selection
  robustly for typical p ≪ n regimes.
- **Datasets that scale with model size** — data-model double
  descent needs the sample count fixed.
- **Highly regularised** models — regularisation smooths out the
  peak (see Nakkiran-Venkat-Kakade-Ma 2021).

## Assumptions & caveats

- **Peak location** depends on parameterisation, task, and
  regulariser; not always exactly at p = n.
- **Data-side double descent** exists too: fix p, sweep n.
- **Epoch-wise double descent**: fixed model, sweep training
  epochs (Nakkiran 2020) — same pattern in training time.
- **Regularisation kills the peak**: with enough ridge / weight
  decay the U-shape re-emerges.

## Related in this repo

- `bias-variance-decomposition`, `regularization-lasso-ridge` —
  classical bias-variance framework.
- `neural-tangent-kernel` — theoretical framework for
  wide-net training.
- `lottery-ticket-hypothesis` — related over-parametrised
  phenomenon.
- `random-projections` — high-dimensional linear-algebra cousin.

## Run

```
python techniques/double-descent/python/double_descent.py
Rscript techniques/double-descent/r/double_descent.R
```

**Refs:** Belkin, M., Hsu, D., Ma, S. & Mandal, S. "Reconciling modern machine-learning practice and the classical bias-variance trade-off." *PNAS* 116(32), 2019; Nakkiran, P. et al. "Deep double descent: Where bigger models and more data hurt." *ICLR*, 2020; Hastie, T., Montanari, A., Rosset, S. & Tibshirani, R. J. "Surprises in high-dimensional ridgeless least squares interpolation." *Ann. Stat.* 50(2), 2022.

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

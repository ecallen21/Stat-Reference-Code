# Double / Debiased ML (Reference §15.25)

Chernozhukov et al. (2018). Estimate a low-dim parameter of
interest `θ` while allowing arbitrary ML nuisance estimation with
**Neyman-orthogonal moments** + **K-fold cross-fitting** to remove
regularisation bias.

## Partially Linear Model (PLR)

```
Y = D · θ + g(X) + ε,    D = m(X) + v
```

1. Split into K folds.
2. On each fold's out-of-fold data, predict `g(X)` and `m(X)`
   with any ML method.
3. Residualise: `ỹ = Y − ĝ(X)`, `d̃ = D − m̂(X)`.
4. `θ̂ = mean(d̃ · ỹ) / mean(d̃²)`.

Neyman-orthogonality: the moment's derivative wrt the nuisance is
zero at truth, so ML bias attenuates faster than √n.

## When to use

- **High-dim / nonlinear confounding** where parametric models
  won't fit.
- **Precise causal-effect estimation** with ML flexibility and
  valid inference.

## When NOT to use

- **Weak overlap** — cross-fitting doesn't fix positivity
  violations.
- **Sparse D variation** — small `Var(d̃)` blows the estimator.

## Files

- `python/dml_double_ml.py` — 5-fold DML-PLR with gradient
  boosting nuisances (custom). Demo (n=1000, nonlinear g and m,
  true θ=0.60): **DML θ̂ = 0.631, SE 0.030, 95 % CI (0.573,
  0.689)** — covers truth.
- `r/dml_double_ml.R` — `DoubleML`, `grf::causal_forest`, `hdm`
  (R); `DoubleML`, `econml.dml`, `causalml` (Python).

## Assumptions & caveats

- **Cross-fitting** is required — same-sample nuisance + parameter
  estimation biases inference.
- **K ≥ 2**; K=5 is standard; K=10 for small samples.
- **Nuisance quality** — ML methods should be tuned; poor
  nuisances → wide SE but still asymptotically valid.
- **PLR is one of many DML variants** — IRM (interactive), IIVM
  (instrumental) available in the same framework.

## Related in this repo

- `iptw`, `g-computation`, `aipw-doubly-robust`,
  `tmle-doubly-robust` — the causal-inference stack.
- `debiased-lasso`, `post-selection-inference` — high-dim
  inference cousins.
- `hte-uplift` — CATE via DML.

## Run

```
python techniques/dml-double-ml/python/dml_double_ml.py
Rscript techniques/dml-double-ml/r/dml_double_ml.R
```

**Refs:** Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. "Double/debiased machine learning for treatment and structural parameters." *Econometrics Journal*, 2018.

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

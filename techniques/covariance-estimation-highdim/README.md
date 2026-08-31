# High-Dim Covariance Estimation (Reference §32.11)

When `p` is comparable to `n`, the **sample covariance** has huge
estimation error and is **singular** for `p ≥ n`. Standard remedies:

## Ledoit-Wolf shrinkage (2004)

```
Σ̂_shrink = α · μ̂ I + (1 − α) · S
```

with `μ̂ = tr(S) / p` and `α` chosen analytically to minimise
Frobenius risk. Always well-conditioned; converges to `S` when `n → ∞`.

## Banded / tapered (Bickel-Levina 2008)

For **ordered** features, zero out entries with `|i − j| > bandwidth`.

## Graphical LASSO (see `gaussian-graphical-model`)

Sparse **precision** matrix (inverse covariance), not the covariance
itself.

## When to use

- **Portfolio construction** — Ledoit-Wolf is a standard replacement
  for the sample covariance.
- **Any linear discriminant / QDA / Mahalanobis** where the sample
  cov is ill-conditioned.
- **Time-series ordered features** — banded / tapered.
- **Genomics / metabolomics** where sparsity of the precision is
  natural.

## When NOT to use

- **`n ≫ p`** — the sample covariance is fine.
- **No natural ordering** — banded / tapered inapplicable; use
  Ledoit-Wolf or GLASSO.

## Files

- `python/covariance_estimation_highdim.py` — from-scratch Ledoit-Wolf
  + banded covariance. Demo AR(1)-covariance truth, `p=40, n=30`:
  **sample cov Frobenius error 6.79 → LW 6.15 → banded (bw=2) 5.14**;
  LW shrinkage intensity α = 0.454.
- `r/covariance_estimation_highdim.R` — `corpcor`, `CovTools`,
  `glasso`, `huge`, `spcov` (R); `sklearn.covariance.LedoitWolf / OAS`
  (Python).

## Assumptions & caveats

- **Ledoit-Wolf target** — diagonal-shrinkage is the classical
  version; block-shrinkage variants are richer.
- **Bandwidth choice** — cross-validation on log-likelihood.
- **Frobenius risk vs spectral risk** — different targets give
  different `α`; Ledoit-Wolf 2004 minimises Frobenius, Won-Lim-Kim-
  Rajaratnam 2013 minimises spectral.
- **Non-Gaussian data** — the shrinkage intensity is agnostic; the
  target may not be.
- **Positive definiteness** — LW is guaranteed PD; banded is not
  (fix with nearest-PD projection).

## Related in this repo

- `gaussian-graphical-model` — sparse precision matrix.
- `probabilistic-pca`, `sparse-pca` — factor-based covariance.
- `sandwich-robust-se` — Huber-White robust variance estimator.
- `ledoit-wolf-shrinkage` — actual sklearn's estimator we mirror.

## Run

```
python techniques/covariance-estimation-highdim/python/covariance_estimation_highdim.py
Rscript techniques/covariance-estimation-highdim/r/covariance_estimation_highdim.R
```

**Refs:** Ledoit, O. & Wolf, M. "A well-conditioned estimator for large-dimensional covariance matrices." *JMVA*, 2004; Bickel, P.J. & Levina, E. "Regularized estimation of large covariance matrices." *Annals of Statistics*, 2008.

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

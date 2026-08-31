# Sliced Inverse Regression / Sufficient Dim Reduction (Reference §25.3)

Li (1991) — supervised dimension reduction that finds a **central
subspace** `B` such that `Y ⊥⊥ X | Bᵀ X`. Directions of the central
subspace summarise the regression dependence between `Y` and `X`.

## SIR algorithm

1. Standardise `X`: `Σ̂^(-1/2) (X − μ) = Z`.
2. Slice `Y` into `H` equal-count bins; compute `𝔼[Z | slice_h] = z̄_h`.
3. Form the moment matrix `M = Σ_h p_h · z̄_h z̄_hᵀ`.
4. Take top-`k` eigenvectors of `M`; back-transform by `Σ̂^(-1/2)`.

`k` is the estimated dimension of the central subspace; often chosen
by a permutation test or scree plot.

## When to use

- **Supervised dimension reduction** — you have `Y` and want to
  visualise / regress in a lower-dim space.
- **Single-index or few-index models** — SIR recovers directions
  faster than nonlinear autoencoders.
- **Interpretability** — the directions are linear combinations of
  original features.

## When NOT to use

- **Symmetric dependence on Y** — SIR fails; use SAVE (slice
  covariances) or SIR-II (Cook 1998).
- **Non-elliptical X** — the linearity condition may fail; PHd or
  MAVE better.
- **Highly non-linear / non-smooth signals** — deep models handle them
  better.

## Files

- `python/sir_sufficient_dim_reduction.py` — from-scratch: standardise,
  slice, moment matrix, eigendecomposition, back-transform. Demo on
  single-index truth `y = sin(3 Xβ)` with `p = 5`, `n = 800`:
  **direction alignment |cos| = 0.95** (true direction well-recovered).
- `r/sir_sufficient_dim_reduction.R` — `dr`, `edrGraphicalTools`, `ldr`
  (R); `sliced` (Python).

## Assumptions & caveats

- **Number of slices `H`** — 10-20 is typical; too few = noisy `M`,
  too many = tiny slice means.
- **Elliptical `X` assumption** — the linearity condition holds
  automatically for MVN; otherwise re-standardise or use kernel
  extensions.
- **Symmetric dependence pathology** — quadratic-in-`z` dependencies
  are missed by SIR; use SAVE.
- **Dimension `k`** — chi-square permutation test (Cook 2004) or
  cross-validated regression error.
- **Standard errors** — asymptotic (Zhu-Fang 1996) or bootstrap.

## Related in this repo

- `dimensionality-reduction-pca` (if present) / `sparse-pca` /
  `kernel-pca` — unsupervised alternatives.
- `single-index-model` — SIR extended to unknown link + β together.
- `partial-least-squares` / `canonical-correlation` — supervised
  covariance-based DR.
- `varying-coefficient-model` — a semiparametric alternative when
  effects vary with a modifier.

## Run

```
python techniques/sir-sufficient-dim-reduction/python/sir_sufficient_dim_reduction.py
Rscript techniques/sir-sufficient-dim-reduction/r/sir_sufficient_dim_reduction.R
```

**Refs:** Li, K.-C. "Sliced inverse regression for dimension reduction." *JASA*, 1991; Cook, R.D. & Weisberg, S. "Discussion of sliced inverse regression for dimension reduction (SAVE)." *JASA*, 1991; Cook, R.D. *Regression Graphics*, Wiley, 1998.

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

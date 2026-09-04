# Seemingly Unrelated Regression (Reference §35.7)

Zellner (1962). Fit **M equations sharing no common regressors but
correlated errors**:

```
y_i = X_i β_i + ε_i,      Cov(ε_i, ε_j) = σ_ij · I,   i, j ∈ 1..M.
```

## Feasible GLS (Zellner)

1. OLS on each equation → residuals `ê_i`.
2. `Σ̂_ij = ê_iᵀ ê_j / n`.
3. Stack the system and apply GLS with weight `Σ̂⁻¹ ⊗ I_n`.

**SUR = OLS** if regressors are identical across equations or `Σ` is
diagonal. **Efficiency gain** grows with cross-equation error
correlation.

## When to use

- **Multiple related regressions** — demand systems, factor models,
  multiple-outcome trials.
- **Correlated errors** across outcomes.
- **Testing cross-equation restrictions** (Wald on stacked
  coefficient vector).

## When NOT to use

- **Same regressors in every equation** — reduces to OLS.
- **Diagonal error covariance** — reduces to OLS.
- **Very large M** — the `M × M` Σ estimate becomes noisy; consider
  shrinkage (Ledoit-Wolf).

## Files

- `python/sur_regression.py` — Zellner FGLS on 2-equation system with
  `Cov(ε_1, ε_2) = 0.8`. Demo:
  - **Σ_hat close to true** (0.976, 0.766, 1.003 vs true 1.0, 0.8, 1.0).
  - β̂ from SUR and OLS both close to truth; SUR shrinks noise via
    cross-equation info-sharing.
- `r/sur_regression.R` — `systemfit::systemfit` (R reference);
  `linearmodels.system.SUR` (Python).

## Assumptions & caveats

- **Small-sample bias** of Σ̂ — can push SUR further from truth than
  OLS in small n; use shrinkage.
- **Non-linear SUR** (nonlinear SUR / nSUR) generalises via GMM.
- **3SLS** — SUR + IV for systems with endogenous regressors.
- **Cross-equation restrictions** — impose during GLS fit for tests.

## Related in this repo

- `iv-2sls` — instrument-variable analogue.
- `gmm-general` — SUR is a special case.
- `heckman-selection` — sample-selection system.
- `nonlinear-least-squares` — non-linear analogue.

## Run

```
python techniques/sur-regression/python/sur_regression.py
Rscript techniques/sur-regression/r/sur_regression.R
```

**Refs:** Zellner, A. "An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias." *JASA*, 1962; Greene, W.H. *Econometric Analysis*, 8th ed., Pearson, 2018 (Ch. 10).

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

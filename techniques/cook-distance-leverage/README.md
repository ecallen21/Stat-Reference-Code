# Cook's Distance / Leverage (Reference §47.333)

Cook (1977). Regression influence diagnostics:

```
h_ii = X (XᵀX)⁻¹ Xᵀ[i, i]                        leverage
D_i  = r_i² / (p · (1 − h_ii)) · h_ii / (1 − h_ii)   Cook's D
```

Rules of thumb: `h_ii > 2p/n` flags high leverage; `D_i > 4/n`
flags influential; `D_i > 1` flags highly influential.

## Files

- `python/cook_distance_leverage.py` — Direct diagonal-of-hat
  matrix + studentised residuals + D. Toy regression n=40,
  p=3 with injected high-leverage x-outlier (idx 0) and two
  response outliers (idx 5, 10). Top-D points correctly flag
  the injected outliers.
- `r/cook_distance_leverage.R` — `stats::influence.measures`,
  `car::influenceIndexPlot`, `performance::check_outliers` (R);
  `statsmodels.stats.outliers_influence.OLSInfluence`,
  `yellowbrick.regressor.CooksDistance`, from-scratch (Python).

## When to use

- **Regression diagnostics** — always inspect Cook's D and
  leverage.
- **Detecting data-entry errors** — high-D points are the
  first suspects.
- **Model validation** — refit without top-D points and
  compare.

## When NOT to use

- **Non-linear models** — use case-deletion versions
  (jackknife) or influence functions.
- **Very large n** — full hat matrix is O(n²); use recursive
  QR or leverage-only score.
- **Correlated errors** — Cook's D assumes independent
  Gaussian errors; adjust for cluster / time-series.

## Assumptions & caveats

- **Rank-deficient X** — regularise before computing h_ii.
- **DFBETAS complement D** — per-coefficient influence.
- **Robust residuals** — externally studentised residuals
  (deletion residuals) preferable to internal.
- **Multiple influential points** — masking / swamping;
  iterative deletion may help.

## Related in this repo

- `regression-diagnostics` — the umbrella tool.
- `sandwich-robust-se` — variance corrections.
- `influence-functions-eif` — modern parametric analogue.
- `robust-regression`, `mm-estimators-robust` — robust
  regression alternatives.

## Run

```
python techniques/cook-distance-leverage/python/cook_distance_leverage.py
Rscript techniques/cook-distance-leverage/r/cook_distance_leverage.R
```

**Refs:** Cook, R.D. "Detection of influential observations in linear regression." *Technometrics*, 19(1): 15-18, 1977; Belsley, D.A., Kuh, E. and Welsch, R.E. *Regression Diagnostics*, Wiley, 1980.

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

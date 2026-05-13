# Regression Diagnostics (Reference §5.6, §5.30, §5.39)

After OLS, check the assumptions and identify problematic observations. All the quantities here are derivable from the **hat matrix** `H = X(X'X)⁻¹X'` and the residuals.

## Per-observation diagnostics

| Quantity | Formula | What it tells you |
|----------|---------|-------------------|
| **Leverage** `hᵢ` | `H_{ii}` | How extreme `xᵢ` is in the predictor space. Σ`hᵢ = p`; range [0, 1] |
| Raw residual `eᵢ` | `yᵢ − ŷᵢ` | Just the error |
| **Standardized residual** `sᵢ` | `eᵢ / (σ̂ · √(1−hᵢ))` | Residual on a unit-variance scale |
| **Studentized (deleted) residual** `tᵢ` | `eᵢ / (σ̂₍ᵢ₎ · √(1−hᵢ))` | Same idea but σ̂ is estimated *without* obs i; ~`t_{n−p−1}` under the null |
| **Cook's distance** `Dᵢ` | `(sᵢ² / p) · hᵢ/(1−hᵢ)` | Overall influence on the fitted values |
| **DFFITS** `DFFITSᵢ` | `tᵢ · √(hᵢ/(1−hᵢ))` | Scaled change in `ŷᵢ` when obs i is dropped |
| **DFBETAS** `DFBETASᵢⱼ` | scaled change in `β̂_j` when obs i is dropped | Per-coefficient influence |

## Rules of thumb (Belsley, Kuh & Welsch 1980)

| Diagnostic | Threshold |
|------------|-----------|
| High leverage | `hᵢ > 2p/n` (some authors use `3p/n`) |
| Large Cook's D | `Dᵢ > 4/n` |
| Large DFFITS | `\|DFFITSᵢ\| > 2√(p/n)` |
| Large DFBETAS | `\|DFBETASᵢⱼ\| > 2/√n` |

## High leverage ≠ influential

- **High leverage** = `xᵢ` is extreme in predictor space; obs i has the *potential* to influence the fit.
- **High influence** = obs i actually *does* change the fit (large Cook's D / DFFITS).
- A high-leverage point with a small residual is fine; a low-leverage point with a huge residual is an outlier but not a force-mover; high leverage *plus* a large residual is the dangerous case.

## What to do about a flagged point

1. **Look at the actual observation.** Is it a data error? A clerical mistake? A genuinely different unit?
2. **Refit without it.** If the conclusions are robust, note both fits in the report. If they flip, you have a finding *about* that observation, not about the population.
3. **Don't auto-delete.** "It had Cook's D > 4/n so I removed it" is not a defensible cleaning step — see the caveats in `techniques/outlier-tests`.
4. Use a **robust regression** (`techniques/robust-regression`) if you have many such points.

## Files
- `python/regression_diagnostics.py` — from-scratch hat matrix, leverage, standardized + studentized residuals, Cook's D, DFFITS, DFBETAS, with thresholds; matches `statsmodels.OLSResults.get_influence()`.
- `r/regression_diagnostics.R` — from-scratch + base `hatvalues`, `cooks.distance`, `dffits`, `dfbetas`, `rstandard`, `rstudent`.
- PySpark: N/A — diagnostics depend on `(X'X)⁻¹` and per-row `hᵢ`; for moderate `p` you can `toPandas()` the fitted residuals and use the Python version. For massive data, use Spark for the fit and run diagnostics on a sub-sample.

## Run
```
python techniques/regression-diagnostics/python/regression_diagnostics.py
Rscript techniques/regression-diagnostics/r/regression_diagnostics.R
```

**Refs:** Belsley, Kuh & Welsch, *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*, Wiley, 1980; Cook & Weisberg, *Residuals and Influence in Regression*, Chapman & Hall, 1982.

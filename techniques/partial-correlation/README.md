# Partial & Semi-Partial Correlation (Reference §4.5)

Both ask: **what's the association between `x` and `y` once we account for the linear effect of one or more covariates `Z`?**

## Partial correlation `r_{xy . Z}`

Residualize both `x` and `y` on `Z` (via OLS), then correlate the residuals. For a single covariate `z`:
`r_{xy.z} = (r_{xy} − r_{xz} · r_{yz}) / √((1 − r_{xz}²)(1 − r_{yz}²))`.

For multiple covariates, the residual definition still works; equivalently, all pairwise partials are read off the **precision matrix** `P = inv(Σ)`:
`r_{ij | rest} = −P_{ij} / √(P_{ii} · P_{jj})`.

## Semi-partial (part) correlation `r_{x(y.Z)}`

Residualize **only one side** on `Z`, then correlate. With `y` residualized (the usual convention):
`r_{x(y.z)} = (r_{xy} − r_{xz} · r_{yz}) / √(1 − r_{yz}²)`.

`r_{x(y.Z)}²` is the **unique R²** contribution of `x` in a model that already has `Z` — exactly the partial-F change-in-R² you'd compute in regression.

## When to use which

- **Partial** — symmetric in `x` and `y`; "association between `x` and `y` after removing the shared linear influence of `Z`." Classic confounder adjustment.
- **Semi-partial** — asymmetric; "how much *additional* explanatory power does `x` provide for `y` beyond `Z`?" Aligned with regression coefficient interpretation.

## Significance test

`t = r · √((n − 2 − p) / (1 − r²))` ~ `t_{n − 2 − p}`, where `p = |Z|`. Same form for partial and semi-partial (df identical).

## Watch out

- All of this is *linear* — a nonlinear effect of `Z` won't be removed by linear residualization, so a partial r near zero doesn't prove independence.
- Conditioning on a **collider** (a variable causally affected by both `x` and `y`) can *create* spurious partial correlation. See `techniques/causal-inference` in a future batch.

## Files
- `python/partial_correlation.py` — from-scratch via OLS residualization for both partial and semi-partial; plus a `partial_correlation_matrix` helper that returns all pairwise partials from the precision matrix; compares against `pingouin.partial_corr` (optional).
- `r/partial_correlation.R` — from-scratch + `ppcor::pcor.test` / `ppcor::spcor.test`.
- PySpark: N/A — pull a sub-sample (`df.sample(...).toPandas()`) and use the Python version; for very large `Z`, fit `lm` in Spark MLlib and compute residuals there, then `F.corr` on the residual columns.

## Run
```
python techniques/partial-correlation/python/partial_correlation.py
Rscript techniques/partial-correlation/r/partial_correlation.R
```

**Ref:** Cohen, Cohen, West & Aiken, *Applied Multiple Regression/Correlation Analysis for the Behavioral Sciences*, 3rd ed., 2003.

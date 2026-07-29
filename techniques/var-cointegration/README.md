# VAR + Cointegration + Error Correction (Reference §13.12, §13.13, §13.44)

Three tightly-related tools for **multivariate** time series:

## VAR — Vector Autoregression (§13.12)

Each series regressed on lags of all K series:

```
y_t  =  c  +  A_1 y_{t-1}  +  A_2 y_{t-2}  +  ...  +  A_p y_{t-p}  +  u_t
```

Fitted equation-by-equation via OLS (efficient because each equation shares the same regressors). Order `p` picked by AIC / BIC.

## Cointegration (§13.13)

Two (or more) I(1) series are **cointegrated** if some linear combination is stationary. Meaning: they share a long-run equilibrium — like a couple walking a dog on a leash. Each wanders (non-stationary), but the distance between them stays bounded (stationary).

## Engle-Granger 2-step test (§13.44)

1. OLS `y_1 = α + β y_2 + u`.
2. ADF on residuals `u`.

Small p → residuals stationary → series cointegrated with rank 1.

## Error Correction Model (§13.13)

If cointegrated, one series *corrects* toward the long-run relation:

```
Δy_1_t  =  c  +  γ · (y_1_{t-1} − β · y_2_{t-1})  +  φ · Δy_1_{t-1}  +  ψ · Δy_2_{t-1}  +  ε
              ↑ error-correction term ↑
```

`γ < 0` means `y_1` corrects back toward the equilibrium; the size of `γ` is the **speed of adjustment** (larger absolute γ = faster).

## Files

- `python/var_cointegration.py` — from-scratch VAR via equation-by-equation OLS + Engle-Granger 2-step + ECM. Cointegrating β = 1.98 vs. true 2.0 on the demo; ECM γ = -0.66 (correcting back to equilibrium).
- `r/var_cointegration.R` — thin wrappers around `vars::VAR` + `tseries::po.test` (Phillips-Ouliaris) + `urca::ca.jo` (Johansen — more powerful multivariate cointegration test).

## Assumptions

- **VAR** — stationarity of the modeled series (difference first if non-stationary).
- **Cointegration** — the input series are individually I(1) (unit-root non-stationary but stationary after one differencing).
- **ECM** — presupposes cointegration exists.

## When to prefer Johansen over Engle-Granger

- Multiple cointegrating relations possible.
- Symmetric treatment of the series (EG-2 is asymmetric — you pick which is LHS).
- Better small-sample properties.

## Run

```
python techniques/var-cointegration/python/var_cointegration.py
Rscript techniques/var-cointegration/r/var_cointegration.R
```

**Refs:** Sims, C.A. "Macroeconomics and reality." *Econometrica* 48(1), 1–48, 1980; Engle, R.F. & Granger, C.W.J. "Co-integration and error correction: representation, estimation, and testing." *Econometrica* 55(2), 251–276, 1987; Johansen, S. "Statistical analysis of cointegration vectors." *J. Econ. Dyn. Control* 12(2–3), 231–254, 1988; Lütkepohl, H. *New Introduction to Multiple Time Series Analysis*, Springer, 2005.

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

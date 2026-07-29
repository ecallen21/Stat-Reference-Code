# GARCH — Generalized Autoregressive Conditional Heteroscedasticity (Reference §13.11; also covers §13.33)

Financial return series show **volatility clustering** — large moves cluster in time. GARCH models the conditional variance `σ_t²` as a recursion:

```
r_t     =  μ + ε_t                ε_t = σ_t · z_t,  z_t ~ N(0, 1)
σ_t²    =  ω  +  α · ε_{t-1}²  +  β · σ_{t-1}²
```

Constraints for a valid stationary model:
- `ω > 0`
- `α ≥ 0`
- `β ≥ 0`
- **`α + β < 1`** (persistence must be less than 1)

Fitted by MLE on the normal-density log-likelihood.

## Persistence

`α + β` is the persistence of shocks — how quickly a volatility burst decays. Values near 1 are typical for daily equity returns (~0.98); the near-non-stationarity implies volatility shocks last a long time.

## §13.33 Multivariate GARCH

Extends volatility modeling to K assets simultaneously:

- **CCC** (Bollerslev 1990) — Constant Conditional Correlation. Simplest; correlations don't move.
- **DCC** (Engle 2002) — Dynamic Conditional Correlation. Correlations follow their own GARCH-like recursion.
- **BEKK** (Engle-Kroner 1995) — full K×K covariance dynamics with positivity by construction. Expensive to fit for large K.

Available in R's `rmgarch` (DCC, GO-GARCH) and Python's `mgarch`. Not implemented from scratch here.

## Files

- `python/garch.py` — from-scratch GARCH(1,1) with normal innovations via BFGS MLE (reparameterized to enforce `ω, α, β > 0` and `α + β < 1`). On synthetic data with true `(0.05, 0.10, 0.85)`, recovers `(0.10, 0.12, 0.77)` and unconditional variance 0.94 vs. true 1.00.
- `r/garch.R` — thin wrapper around `rugarch::ugarchspec` + `rugarch::ugarchfit`.

## Assumptions

- Returns have zero autocorrelation but non-zero squared-return autocorrelation (typical for financial returns).
- Normal innovations by default — real returns often have fatter tails; use t-innovations (`dist="std"` in `rugarch`) or EGARCH for asymmetry (leverage effects).
- Enough data — GARCH is notoriously sample-hungry; typically want ≥ 500 daily observations.

## Run

```
python techniques/garch/python/garch.py
Rscript techniques/garch/r/garch.R
```

**Refs:** Engle, R.F. "Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation." *Econometrica* 50(4), 987–1007, 1982; Bollerslev, T. "Generalized autoregressive conditional heteroskedasticity." *J. Econom.* 31(3), 307–327, 1986; Engle, R.F. "Dynamic conditional correlation." *JBES* 20(3), 339–350, 2002; Ghalanos, A. "Introduction to the rugarch package." *R vignette*, 2022.

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

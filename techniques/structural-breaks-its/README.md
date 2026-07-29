# Structural Breaks + Interrupted Time Series (Reference §13.7, §13.10)

## Chow test (§13.10) — known break time

Split the series at `t*`. Compare pooled SSR to sum of half-SSRs:

```
F  =  ((SSR_pooled − SSR_1 − SSR_2) / k)  /  ((SSR_1 + SSR_2) / (n − 2k))
```

with F(k, n − 2k) under H₀ of no break. Small p → structural break at t*.

## Bai-Perron single-break scan (§13.10)

Try every candidate `t*`; report the maximum-F break. Proper critical values for the sup-F test need simulation (Andrews 1993; Bai-Perron 2003); this file reports the raw F p-value at the best break as a lower bound. For rigorous inference use `strucchange::breakpoints` in R.

## Interrupted Time Series (§13.7)

Regression with an intervention at known time `t*`, allowing both a **level** shift (`b_2`) and a **slope** change (`b_3`):

```
y_t  =  b_0  +  b_1 · t  +  b_2 · D_t  +  b_3 · (t − t*) · D_t  +  ε_t
      D_t = 1 if t ≥ t*, else 0
```

- `b_1`: pre-intervention trend
- `b_2`: instantaneous level change at intervention
- `b_3`: change in slope after intervention

The classical evaluation design for policy interventions (Bernal, Cummins & Gasparrini 2017).

## Files

- `python/structural_breaks_its.py` — Chow test at known break + max-F scan + ITS regression. Bai-Perron correctly locates a synthetic break at t=50; ITS recovers all four coefficients.
- `r/structural_breaks_its.R` — thin wrappers around `strucchange::sctest` (Chow) + `strucchange::breakpoints` (Bai-Perron, with proper critical values) + `lm()` for ITS.

## Assumptions

- **Chow / Bai-Perron**: linear model within each regime; homoscedastic residuals within each regime.
- **ITS**: no autocorrelation in residuals (test Ljung-Box on residuals; if violated, fit an ARIMA error structure via SARIMAX or GLS).
- Regime boundaries either known (Chow, ITS) or one at a time (single-break scan).

## Run

```
python techniques/structural-breaks-its/python/structural_breaks_its.py
Rscript techniques/structural-breaks-its/r/structural_breaks_its.R
```

**Refs:** Chow, G.C. "Tests of equality between sets of coefficients in two linear regressions." *Econometrica* 28(3), 591–605, 1960; Bai, J. & Perron, P. "Computation and analysis of multiple structural change models." *J. Appl. Econom.* 18(1), 1–22, 2003; Bernal, J.L., Cummins, S. & Gasparrini, A. "Interrupted time series regression for the evaluation of public health interventions: a tutorial." *IJE* 46(1), 348–355, 2017.

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

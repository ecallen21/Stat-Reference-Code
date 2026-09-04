# Panel Cointegration (Reference §35.26)

Pedroni (1999, 2004), Westerlund (2007), Pesaran-Shin-Smith PMG
(1999). For a **panel of I(1) series**, test whether `y_it` and `x_it`
share a **long-run equilibrium** via cointegrated residuals.

## Pedroni residual approach

```
y_it = α_i + β_i x_it + e_it                (unit-by-unit cointegrating regression)
apply ADF / Phillips-Perron to residuals e_it, pool the statistic.
```

Variants:
- **Panel PP / ADF** — group-mean or panel-level pooling.
- **Panel variance ratio**.
- **Westerlund ECM** — testing whether the error-correction term is
  significant.

## PMG (Pesaran-Shin-Smith)

Pool long-run coefficients while allowing short-run heterogeneity —
useful when the long run is common but adjustment differs across units.

## When to use

- **Long-run relationships in macro panels** — cross-country growth,
  purchasing-power parity, income convergence.
- **Panel with I(1) variables** — test cointegration before regressing.
- **Firm-level dynamics** with common trends.

## When NOT to use

- **Small T** — unit-root / cointegration tests are asymptotic in T.
- **Cross-sectional dependence** — Pedroni assumes independence across
  units; use Westerlund with bootstrap p-values instead.

## Files

- `python/panel_cointegration.py` — Pedroni residual ADF with
  group-mean pooling. Demo (N=30, T=60):
  - **Cointegrated** (`y = 0.7 x + stationary noise`): group-mean
    ADF t = −8.19 (reject no cointegration).
  - **Not cointegrated** (two independent random walks): group-mean
    ADF t = −1.93 (fail to reject).
- `r/panel_cointegration.R` — `plm::pcointtest`, `punitroots`, `urca`
  (R); `arch.unitroot`, `egcm`, `linearmodels` (Python).

## Assumptions & caveats

- **Critical values** are from Pedroni tables; the demo shows
  relative magnitudes.
- **Cross-sectional dependence** — Pedroni relies on independence;
  correct with cross-sectional demeaning or block bootstrap.
- **Panel unit root** first (IPS / CIPS / Fisher-ADF) to check
  whether variables are I(1).
- **Heterogeneous cointegrating vectors** — the group-mean approach
  allows β_i to vary; the panel-level pools β.
- **Short T** — statistics are asymptotic; small samples require
  bootstrap.

## Related in this repo

- `var-cointegration` — single-series Johansen / Engle-Granger.
- `fixed-effects-panel`, `arellano-bond-gmm`, `hausman-test` — panel
  workhorses.
- `newey-west-hac` — SEs for regressions with autocorrelated errors.
- `time-series-forecasting` — related dynamic-model machinery.

## Run

```
python techniques/panel-cointegration/python/panel_cointegration.py
Rscript techniques/panel-cointegration/r/panel_cointegration.R
```

**Refs:** Pedroni, P. "Critical values for cointegration tests in heterogeneous panels with multiple regressors." *Oxford Bulletin of Economics and Statistics*, 1999; Westerlund, J. "Testing for error correction in panel data." *Oxford Bulletin of Economics and Statistics*, 2007; Pesaran, M.H., Shin, Y. & Smith, R.P. "Pooled mean group estimation of dynamic heterogeneous panels." *JASA*, 1999.

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

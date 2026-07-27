# Cox Model Residuals + PH Assumption Test (Reference §11.33; also covers §11.53)

Four residual types + one formal test for proportional-hazards violation.

## Residuals

| Residual | Formula | Diagnostic for |
|---|---|---|
| **Schoenfeld** | `X_j − E[X | R(t_j), event]` (one per event, per covariate) | Proportional hazards |
| **Scaled Schoenfeld** | `d · cov(β̂) · r_S + β̂` | Basis for the Grambsch-Therneau PH test |
| **Martingale** | `event_i − Ĥ(t_i | X_i)` | Functional form of a covariate |
| **Cox-Snell** | `Ĥ(t_i | X_i)` | Overall model fit (should be Exp(1) if OK) |
| **Deviance** | `sign(r_M) · √(−2(r_M + e·log(e − r_M)))` | Outlier detection |

## Grambsch-Therneau PH test

Correlate scaled Schoenfeld residuals with a monotone function `g(t)` (log-t or rank-t typically). A significant correlation for covariate k means the log-HR is drifting with time → PH violated for that covariate. Per-covariate and global χ² tests provided.

## What to do if PH is violated

- **Stratify** on the offending covariate (baseline hazard varies by stratum; no HR estimated for it).
- **Include a time-interaction term** `X_k · f(t)` — turns the model into an extended Cox.
- **Switch away from HR-based inference** — use RMST or an AFT model for a time-scale summary.

## Files

- `python/cox_diagnostics.py` — all five residual types + the Grambsch-Therneau test; reuses `fit_cox` from [`cox-ph`](../cox-ph).
- `r/cox_diagnostics.R` — thin wrapper around `survival::residuals()` and `survival::cox.zph()`.

## Assumptions

- Cox model has been fit and its results are needed for diagnosis (not an alternative fit).
- Enough events for the χ² approximations (rule of thumb: `n_events ≥ 20 · p`).

## Run

```
python techniques/cox-diagnostics/python/cox_diagnostics.py
Rscript techniques/cox-diagnostics/r/cox_diagnostics.R
```

**Refs:** Schoenfeld, D. "Partial residuals for the proportional hazards regression model." *Biometrika* 69(1), 239–241, 1982; Grambsch, P.M. & Therneau, T.M. "Proportional hazards tests and diagnostics based on weighted residuals." *Biometrika* 81(3), 515–526, 1994; Cox, D.R. & Snell, E.J. "A general definition of residuals." *JRSS B* 30(2), 248–275, 1968; Therneau, T.M. & Grambsch, P.M. *Modeling Survival Data: Extending the Cox Model*, Springer, 2000.

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

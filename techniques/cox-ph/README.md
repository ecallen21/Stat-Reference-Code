# Cox Proportional-Hazards Model (Reference §11.8; also covers §11.16, §11.42, §11.54, §11.59, §11.63, §11.64, §11.66)

The **workhorse** survival regression model:

```
h(t | X)  =  h₀(t) · exp(X·β)
```

So the hazard ratio between two subjects at the same t is `exp((X_1 − X_2)·β)` — the baseline `h₀(t)` cancels. The model is **semi-parametric**: parametric in `X` but non-parametric in the baseline hazard.

## Fitting — Partial Likelihood

```
PL(β)  =  ∏_{events t_j}  exp(X_j·β) / Σ_{k ∈ R(t_j)} exp(X_k·β)
```

Maximize via Newton-Raphson on the log-partial-likelihood; the Hessian gives asymptotic SEs.

## Ties

Two tied-event corrections implemented:

- **Efron** (default) — better small-sample behavior; adjusts the denominator sequentially for each of the d_j tied events.
- **Breslow** — the crude version; simpler, adequate when ties are rare.

## Left truncation and time-varying covariates

Handled uniformly via the **counting-process input** `(start, stop, event, X)`:

- **§11.42 / §11.59 Left truncation (delayed entry)**: pass `start > 0` — subjects only enter the risk set at their entry time. Standard for late-entry cohorts.
- **§11.16 / §11.54 Time-varying covariates**: split each subject into multiple `(start, stop, X)` rows whenever `X` changes. The likelihood formula is unchanged; risk sets are computed on `(start, stop]` intervals.

## Also covered

- **§11.63** HR interpretation: `HR = 2.0` means "double the instantaneous hazard, all else equal" — it does **not** mean "halve the survival time." For time-scale effects use `rmst` or an AFT model.
- **§11.64** Events-per-Variable (EPV) rule: `epv_rule_of_thumb(n_events, p)` returns EPV; traditional floor is 10, aim for ≥ 20 (Riley et al. 2019).
- **§11.66** Sample size for Cox = sample size for the log-rank test: `d ≈ (z_{1−α/2} + z_{power})² / (log HR)² / p(1−p)` where `p` is randomization proportion. Not shipped as a helper here — trivial once you know it.

## Files

- `python/cox_ph.py` — partial-likelihood Newton-Raphson from scratch (Efron + Breslow ties), left truncation, counting-process input, EPV helper. Beta / SE / HR / z / p match `lifelines.CoxPHFitter` when installed (env here lacks it — cross-check theoretical from-scratch demo).
- `r/cox_ph.R` — thin wrapper around `survival::coxph`, which is the authoritative Cox implementation.

## Assumptions

- **Proportional hazards**: `HR` is constant over time. Test with [`cox-diagnostics`](../cox-diagnostics) (Grambsch–Therneau on scaled Schoenfeld residuals). If violated: stratify, use time-varying covariates, or switch to RMST / AFT.
- **Independent censoring**.
- **Linearity in log-hazard**: for continuous covariates, check martingale residuals; consider splines if non-linear.

## Run

```
python techniques/cox-ph/python/cox_ph.py
Rscript techniques/cox-ph/r/cox_ph.R
```

**Refs:** Cox, D.R. "Regression models and life-tables." *JRSS B* 34(2), 187–220, 1972; Efron, B. "The efficiency of Cox's likelihood function for censored data." *JASA* 72(359), 557–565, 1977; Andersen, P.K. & Gill, R.D. "Cox's regression model for counting processes: a large sample study." *Ann. Stat.* 10(4), 1100–1120, 1982; Riley, R.D., Snell, K.I., Ensor, J. *et al.* "Minimum sample size for developing a multivariable prediction model." *Stat. Med.* 38(7), 1276–1296, 2019.

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

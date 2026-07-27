# Kaplan-Meier Survival Estimator (Reference §11.2; also covers §11.1, §11.45, §11.46, §11.61, §11.68)

Non-parametric estimator of the survival function `S(t) = P(T > t)` from **right-censored** data:

```
Ŝ(t)  =  ∏_{t_j ≤ t}  (1 − d_j / n_j)

d_j = # events at t_j
n_j = # at risk just before t_j
```

Also covered here:

- **§11.1** Life tables — same idea binned into fixed intervals; not built separately.
- **§11.45** IPTW-adjusted KM — pass `weights=` (e.g. inverse-probability-of-treatment weights) to `kaplan_meier()`.
- **§11.46** Median survival time — `median_survival(km)` returns median + Brookmeyer-Crowley 95% CI.
- **§11.61** Time origin — the `times` you pass in defines t = 0. Trial randomization, diagnosis date, symptom onset all yield different survival curves; document your choice.
- **§11.68** Risk-table publication plots — `risk_table(times, events, grid, groups=...)` returns per-group counts at risk at each grid time.

## Variance and confidence intervals

**Greenwood's formula**:
```
Var(Ŝ(t))  =  Ŝ(t)²  ·  Σ_{t_j ≤ t}  d_j / (n_j (n_j − d_j))
```

**Pointwise 95% CI on the log-log scale** (better small-sample coverage than plain Wald):
```
g(S) = log(−log S)
Var(g) = Var(Ŝ) / (Ŝ · log Ŝ)²
CI on g:  g ± z · √Var(g)
CI on S:  exp(−exp(g_hi)), exp(−exp(g_lo))
```

## Median survival CI — Brookmeyer-Crowley (1982)

Invert the sign-based test: the CI is the set of times `t` such that
```
(Ŝ(t) − 0.5)² ≤ z² · Var(Ŝ(t))
```

## Files

- `python/kaplan_meier.py` — KM + Greenwood variance + log-log CI + median + Brookmeyer-Crowley CI + `risk_table()` helper. Median matches theoretical `log(2)/λ` for exponential DGP within CI on the demo.
- `r/kaplan_meier.R` — from-scratch + `survival::survfit` cross-check.

## Assumptions

- **Independent censoring**: censoring time is independent of event time conditional on the observed covariates. If censoring is informative (e.g. sicker patients drop out earlier), KM is biased — use IPCW methods (§11.36; deferred).
- **Left truncation** (delayed entry) is straightforward — use the counting-process (start, stop, event) input format in `cox-ph`; KM extends the same way.
- Non-informative time origin — subjects with different entry times share the same t = 0 semantics.

## Run

```
python techniques/kaplan-meier/python/kaplan_meier.py
Rscript techniques/kaplan-meier/r/kaplan_meier.R
```

**Refs:** Kaplan, E.L. & Meier, P. "Nonparametric estimation from incomplete observations." *JASA* 53(282), 457–481, 1958; Greenwood, M. "The natural duration of cancer." *Reports on Public Health and Medical Subjects* 33, 1–26, 1926; Brookmeyer, R. & Crowley, J. "A confidence interval for the median survival time." *Biometrics* 38(1), 29–41, 1982; Klein, J.P. & Moeschberger, M.L. *Survival Analysis*, 2nd ed., Springer, 2003 (Ch. 4).

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

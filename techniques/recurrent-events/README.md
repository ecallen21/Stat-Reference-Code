# Recurrent-Event Cox Models: AG, PWP, WLW (Reference §11.17, §11.18, §11.19; also covers §11.41, §11.51)

For subjects who can experience the **same event repeatedly** (hospitalizations, infections, seizures), the choice of framework determines what "hazard ratio" means:

| Framework | Risk set | HR interpretation | When |
|---|---|---|---|
| **Andersen-Gill (AG)** (§11.17) | Every event contributes; all subjects at risk between events | Overall event-rate ratio | Independent events, common effect |
| **PWP total-time** (§11.18) | Subject at risk for event k+1 only after event k; time from origin | Conditional on being at k-th event | Event-order matters |
| **PWP gap-time** (§11.18 + §11.41) | Same as PWP-total but time reset at each event | Effect on time-between-events | Time-since-last-event matters |
| **WLW marginal** (§11.19) | Per event-number: ALL subjects at risk marginally | Marginal effect for the k-th event | Report per-event effect sizes |

All four are just different **row constructions + strata** feeding the same counting-process Cox. This file's driver reuses [`cox-ph`](../cox-ph)'s `fit_cox` with `(start, stop, event)` inputs.

## Key SE gotcha

- **AG** assumes independent events given X. If events cluster within subject (they usually do), use **sandwich / cluster-robust** SEs.
- **WLW** always needs robust SEs (subject appears in every event-number model).

## Files

- `python/recurrent_events.py` — helpers to build AG / PWP-gap / WLW row structures + fits via `fit_cox`. AG recovers the true β ≈ 0.5 on a synthetic Poisson-process DGP.
- `r/recurrent_events.R` — thin wrapper around `survival::coxph(Surv(start, stop, event) ~ x + cluster(id))`.

## Assumptions

- **AG**: proportional hazards, events independent given X (relax via cluster-robust SEs).
- **PWP**: event order is meaningful (e.g. first, second, third infection have distinct semantics).
- **WLW**: subjects at risk marginally for every event number — reasonable for "up to K events" analyses.

## Run

```
python techniques/recurrent-events/python/recurrent_events.py
Rscript techniques/recurrent-events/r/recurrent_events.R
```

**Refs:** Andersen, P.K. & Gill, R.D. "Cox's regression model for counting processes: a large sample study." *Ann. Stat.* 10(4), 1100–1120, 1982; Prentice, R.L., Williams, B.J. & Peterson, A.V. "On the regression analysis of multivariate failure time data." *Biometrika* 68(2), 373–379, 1981; Wei, L.J., Lin, D.Y. & Weissfeld, L. "Regression analysis of multivariate incomplete failure time data by modeling marginal distributions." *JASA* 84(408), 1065–1073, 1989; Cook, R.J. & Lawless, J.F. *The Statistical Analysis of Recurrent Events*, Springer, 2007.

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

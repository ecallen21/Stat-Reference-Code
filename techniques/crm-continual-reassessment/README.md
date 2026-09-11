# Continual Reassessment Method (Reference §47.261)

O'Quigley, Pepe & Fisher (1990). Bayesian dose-finding that
updates a working dose-toxicity model after each cohort and
treats the next patient at the dose whose posterior toxicity
is closest to the target rate `p_target`:

```
p(d, a) = p_0(d)^exp(a)              (power model)
a ~ Normal(0, σ²)                    prior
next-dose = argmin_d |E[p(d, a) | data] − p_target|
```

Adapts smoothly, uses ALL data, and estimates the MTD directly
(vs 3+3, which uses only the last cohort). EWOC (Escalation
with Overdose Control) adds a safety constraint.

## Files

- `python/crm_continual_reassessment.py` — grid-posterior over
  the calibration parameter `a`, 5-dose skeleton. Demo: skeleton
  `[0.05, 0.10, 0.20, 0.35, 0.50]`, true `[0.02, 0.08, 0.18,
  0.40, 0.65]`, target 0.25. Ten 3-patient cohorts converge
  to dose 3 (true tox 0.18) — the correct MTD by target
  proximity.
- `r/crm_continual_reassessment.R` — `dfcrm::crm`, `bcrm`,
  `trialr::stan_crm`, `crmPack` (R); `UBCRM`, from-scratch
  (Python).

## When to use

- **Phase-I oncology / rare-disease** where 3+3 wastes
  patients on sub-therapeutic doses.
- **Continuous dose-response** with a monotone toxicity curve
  where a working model is credible.
- **Combined efficacy-toxicity** — extended CRM (EffTox) picks
  the best dose by joint utility.

## When NOT to use

- **Very heterogeneous patients** — a single dose-toxicity
  curve may not apply; use covariate CRM or PK-guided models.
- **Regulatory conservatism** — some agencies still expect
  3+3; use CRM in the appendix / secondary analysis.
- **Poorly-calibrated prior** — with only a handful of doses,
  a mis-specified skeleton dominates; calibrate ESS carefully.

## Assumptions & caveats

- **Model mis-specification** — the power model is a working
  model; the design's operating characteristics need to be
  simulated across a range of true curves.
- **Prior calibration** — the CRM's prior effective sample
  size (ESS) should be small (0.5 - 1.0) to let data drive.
- **Cohort size 1 vs 3** — cohort 1 is faster to converge but
  slower to accrue; cohort 3 is more common practice.
- **Skipping restriction** — most implementations forbid
  skipping an untried dose; check the software default.

## Related in this repo

- `three-plus-three-dose-escalation` — classical alternative.
- `boin-bayesian-optimal-interval` — modern model-assisted
  cousin.
- `response-adaptive-randomization` — Bayesian adaptation in
  later phases.

## Run

```
python techniques/crm-continual-reassessment/python/crm_continual_reassessment.py
Rscript techniques/crm-continual-reassessment/r/crm_continual_reassessment.R
```

**Refs:** O'Quigley, J., Pepe, M. and Fisher, L. "Continual reassessment method: A practical design for phase 1 clinical trials in cancer." *Biometrics*, 46(1): 33-48, 1990.

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

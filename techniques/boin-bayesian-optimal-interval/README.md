# BOIN — Bayesian Optimal Interval (Reference §47.263)

Liu & Yuan (2015). Model-ASSISTED phase-I dose-finding that
combines the simplicity of 3+3 with the efficiency of CRM. At
each cohort, compare the observed DLT rate to a pre-calculated
interval `(λ_e, λ_d)` around `p_target`:

```
p̂ = n_DLT / n_treated at current dose
if p̂ ≤ λ_e:   escalate
if p̂ ≥ λ_d:   de-escalate
otherwise:    stay
```

Default `λ_e = 0.6 p_target`, `λ_d = 1.4 p_target` (minimises
escalation-decision error under H₀). Simpler transition table
than CRM, similar performance.

## Files

- `python/boin_bayesian_optimal_interval.py` — Boundary
  computation + trial simulator with beta-based safety
  exclusion. Demo: true tox `[0.02, 0.08, 0.18, 0.40, 0.65]`,
  target 0.25. Boundaries `λ_e=0.197`, `λ_d=0.298`. Over 1000
  trials MTD picks: dose 3 (true MTD) 52.8% vs 3+3's 47.8%,
  fewer trials selecting the toxic dose 4. Uses full budget of
  30 patients per trial vs 3+3's ~15.
- `r/boin_bayesian_optimal_interval.R` — `BOIN`, `boinet`,
  `trialr::stan_boin`, `dfcrm` (R); `UBCRM` via reticulate,
  from-scratch (Python).

## When to use

- **Modern phase-I** where you want CRM-like performance with
  the operational simplicity of 3+3.
- **Combination therapy** — 2-D BOIN handles two-agent
  interaction cohorts.
- **Small centres** — the boundary table can be laminated
  and used at the bedside.

## When NOT to use

- **Fully model-based inference required** — CRM gives a
  full posterior; BOIN gives a decision.
- **Highly non-monotone toxicity** — BOIN, like CRM, assumes
  monotone dose-response.
- **Very small trials (n < 10)** — even model-assisted
  designs benefit from strong priors.

## Assumptions & caveats

- **Boundaries are pre-calculated** — depend on `p_target`
  and (implicitly) on the equivalence interval
  `(φ_1, φ_2) = (0.6, 1.4) p_target`.
- **Beta safety rule** — some implementations use different
  early-termination thresholds; check package defaults.
- **Isotonic regression** at the end selects the MTD from all
  tried doses, not just the last cohort.
- **Efficacy layer** — extended BOIN (BOIN-ET / uTPI) adds
  binary efficacy.

## Related in this repo

- `crm-continual-reassessment` — the fully model-based
  cousin.
- `three-plus-three-dose-escalation` — the classical
  comparator.

## Run

```
python techniques/boin-bayesian-optimal-interval/python/boin_bayesian_optimal_interval.py
Rscript techniques/boin-bayesian-optimal-interval/r/boin_bayesian_optimal_interval.R
```

**Refs:** Liu, S. and Yuan, Y. "Bayesian optimal interval designs for phase I clinical trials." *J. R. Stat. Soc. Ser. C Appl. Stat.*, 64(3): 507-523, 2015.

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

# Cost-Effectiveness Analysis (CEA) (Reference §44.16)

Drummond et al. (2015); Briggs, Claxton & Sculpher (2006). Compares
a new treatment vs a comparator on cost + effectiveness (QALYs /
life-years gained).

## Key summaries

    ICER          = (C_new − C_ref) / (E_new − E_ref)
    NMB(λ)        = λ · E − C
    INMB(λ)       = λ · (E_new − E_ref) − (C_new − C_ref)
    Pr(CE at λ)   = P(INMB(λ) > 0)      (bootstrap / PSA)

The **cost-effectiveness plane** plots (ΔE, ΔC) with willingness-to-
pay threshold λ as a slope through the origin.

## Files

- `python/cost_effectiveness_analysis.py` — ICER + bootstrap CEAC
  from scratch. Demo (n=120 per arm, cost ref 5k / new 25k, QALY
  3.0 / 3.4): ICER ≈ $38 813 / QALY; CEAC P(CE at $50k) = 0.85,
  P(CE at $30k) = 0.09.
- `r/cost_effectiveness_analysis.R` — `BCEA`, `heemod`, `dampack`,
  `hesim`, `survHE` (R); from-scratch (Python).

## When to use

- **HTA submissions** — NICE, ICER (Boston), CADTH, PBAC.
- **Comparative effectiveness research** — new therapy vs standard
  of care.
- **Reimbursement decisions** — WTP thresholds vary by jurisdiction.

## When NOT to use

- **Non-quantifiable outcomes** — quality-of-life is proxied by
  QALYs; some outcomes (equity, dignity) resist reduction.
- **Very small n** — CEAC bootstrap intervals are wide; consider
  Bayesian priors.
- **Distributional / equity concerns** — extend to distributional
  cost-effectiveness (Cookson 2017).

## Assumptions & caveats

- **QALY validity** — utility elicitation quality varies (EQ-5D,
  time trade-off, standard gamble).
- **Time horizon** — extrapolation beyond trial follow-up requires
  survival modelling; expect wide CIs.
- **Discount rate** — 3-5 % / year for costs + QALYs; sensitivity
  analysis matters.
- **Perspective** — payer vs societal — reports must state and
  match the funder's requirement.
- **ICER instability** — near ΔE = 0 the ratio is undefined; NMB
  / INMB is more stable.

## Related in this repo

- `benefit-risk-mcda`, `decision-curve-analysis` — decision-
  analytic cousins.
- `nonparametric-bootstrap` — CEAC estimator.
- `bayesian-hierarchical-models` — Bayesian CEA in BCEA.
- `survival` techniques for QALY extrapolation.

## Run

```
python techniques/cost-effectiveness-analysis/python/cost_effectiveness_analysis.py
Rscript techniques/cost-effectiveness-analysis/r/cost_effectiveness_analysis.R
```

**Refs:** Drummond, M.F. et al. *Methods for the Economic Evaluation of Health Care Programmes*, 4th ed., OUP, 2015; Briggs, A., Claxton, K. & Sculpher, M. *Decision Modelling for Health Economic Evaluation*, OUP, 2006.

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

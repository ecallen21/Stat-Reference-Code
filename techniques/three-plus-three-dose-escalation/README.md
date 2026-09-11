# 3+3 Dose Escalation (Reference §47.262)

Classical rule-based phase-I dose-finding (Storer 1989 and
earlier folklore). Cohort of 3 patients per dose:

- **0/3 DLTs**: escalate one level
- **1/3 DLTs**: add 3 more; if 1/6 total, escalate; if ≥ 2/6,
  STOP
- **≥ 2/3 DLTs**: STOP

MTD is defined as the highest dose with < 2/6 DLTs. Simple,
transparent, well-understood by regulators — but wasteful of
information and slow. CRM / BOIN are model-based alternatives.

## Files

- `python/three_plus_three_dose_escalation.py` — Rule-based
  simulator. Demo: true tox `[0.02, 0.08, 0.18, 0.40, 0.65]`,
  target 0.25. Over 1000 simulated trials, MTD picks: none
  0.6%, dose 1 6.3%, dose 2 23.6%, dose 3 47.8% (true MTD),
  dose 4 20.7%, dose 5 1.5%. Mean patients per trial: 14.6 —
  half of BOIN's 30 but with more diffuse MTD selection.
- `r/three_plus_three_dose_escalation.R` —
  `dfcrm::threeplusthree`, `clinfun::dose.finding`, custom
  loop (R); `UBCRM`, from-scratch (Python).

## When to use

- **Regulatory expectation** in classical phase-I oncology
  where 3+3 is the historical default.
- **Small centres** with limited biostatistics support — the
  rules can be executed on paper.
- **Combination-therapy benchmark** — 3+3 is often the
  comparator when introducing model-based designs.

## When NOT to use

- **Rare cancers** where every patient is precious — CRM /
  BOIN concentrate patients near the MTD.
- **Continuous dose escalation** — 3+3 forces a discrete
  ladder.
- **Trials with efficacy signal** integrated into
  decision-making — use EffTox or BOIN with efficacy layer.
- **Combinations of multiple agents** — 3+3 does not extend
  naturally to > 1 dimension.

## Assumptions & caveats

- **Monotone toxicity assumption** — 3+3 assumes DLT rate
  increases with dose.
- **Under-shoots true MTD** — the "next dose above" rule
  frequently stops one level below optimal.
- **Ignores prior cohorts** — decisions use only the most
  recent cohort's DLTs.
- **No formal target rate** — MTD emerges from stopping
  rules, not from a target `p_target`.

## Related in this repo

- `crm-continual-reassessment` — Bayesian model-based
  alternative.
- `boin-bayesian-optimal-interval` — modern
  model-assisted design.

## Run

```
python techniques/three-plus-three-dose-escalation/python/three_plus_three_dose_escalation.py
Rscript techniques/three-plus-three-dose-escalation/r/three_plus_three_dose_escalation.R
```

**Refs:** Storer, B.E. "Design and analysis of phase I clinical trials." *Biometrics*, 45(3): 925-937, 1989.

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

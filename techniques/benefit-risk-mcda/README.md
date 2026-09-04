# Benefit-Risk MCDA (Reference §43.10)

Mt-Isa et al. (2014). Multi-Criteria Decision Analysis frameworks
score drugs on **multiple benefits and harms** simultaneously,
normalise, weight, and aggregate.

## Weighted-sum MCDA

```
overall_score(drug) = Σ_j w_j · normalised_score(drug, criterion_j)
```

- Direction: `higher_is_better = True/False` per criterion.
- Weights elicited from clinicians / regulators; sum to 1.

## NNT / NNH / LHH

- **NNT** (Number Needed to Treat) = `1 / absolute risk reduction`.
- **NNH** (Number Needed to Harm) = `1 / absolute risk increase`.
- **LHH** (Likelihood of being Helped vs Harmed) = `NNH / NNT` —
  a compact benefit-risk ratio.

## When to use

- **Regulatory benefit-risk reports** — FDA / EMA benefit-risk
  frameworks.
- **HTA / reimbursement** decisions that must consider multiple
  dimensions.
- **Patient-facing shared decision-making** — LHH is
  interpretable.

## When NOT to use

- **Single-criterion comparisons** — a plain hazard ratio or
  survival difference is enough.
- **Weight elicitation infeasible** — MCDA without justifiable
  weights is arbitrary; state weights explicitly.

## Files

- `python/benefit_risk_mcda.py` — weighted-sum MCDA over
  (efficacy, majorAE, minorAE, cost) + NNT / NNH / LHH. Demo
  (3 drugs): drug B wins (MCDA 0.625) because its lower AE rate
  outweighs its higher cost; drug A vs placebo: NNT = 6.7,
  NNH = 20 → LHH = 3.
- `r/benefit_risk_mcda.R` — `drugCombo`, `BRAT`, custom MCDA (R);
  custom (Python).

## Assumptions & caveats

- **Weight elicitation** dominates conclusions; publish sensitivity
  analyses (uniform, expert-elicited, patient-elicited).
- **Normalisation** — min-max is sensitive to outliers; z-scores
  can be more robust for continuous criteria.
- **Independence** — MCDA assumes criteria are independent; drug
  cost and adverse-event rate may correlate, requiring adjustment.
- **NNT / NNH** are baseline-risk-dependent — report the reference
  population.

## Related in this repo

- `decision-curve-analysis` — clinical utility framework related.
- `disproportionality-signal-detection` — signal input to MCDA.
- `target-trial-emulation` — effect estimates that feed MCDA.

## Run

```
python techniques/benefit-risk-mcda/python/benefit_risk_mcda.py
Rscript techniques/benefit-risk-mcda/r/benefit_risk_mcda.R
```

**Refs:** Mt-Isa, S., Hallgreen, C.E., Wang, N. et al. "Balancing benefit and risk of medicines: a systematic review and classification of available methodologies." *Pharmacoepidemiology and Drug Safety*, 2014; PROTECT Consortium. *Benefit-Risk Methodology Recommendations*, 2014.

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

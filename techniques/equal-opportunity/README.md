# Equal Opportunity (Reference Ch 31 Fairness)

The **weaker cousin of equalized odds** — Hardt, Price & Srebro (2016)
only requires equal **true-positive rate** across groups. If the harm
is "denying opportunity to a qualified person", equalising TPR is often
enough; enforcing equal FPR on top can be over-constraining.

## Definition

```
P( Ŷ = 1 | Y = 1, A = a )    equal for every group a.
```

Summaries:

```
EOpp difference = max_a TPR_a  −  min_a TPR_a
EOpp ratio      = min_a TPR_a  /  max_a TPR_a
```

## Enforcement (Hardt 2016 post-processing)

For each group `a`, pick a threshold `t_a` so that

```
TPR_a( t_a )  =  target_TPR.
```

By construction TPR is equalised; FPRs are left free to differ.

## When to use

- **Approval-style decisions** where the harm is missing a qualified
  applicant (loans, admissions, hiring).
- **Screening / diagnosis** — miss a true case is worse than a false
  alarm.
- **Auditing a deployed classifier** — single-number report:
  `EOpp diff`.

## When NOT to use

- **False positives carry material harm** (arrest recommendations,
  medical over-treatment) — use `equalized-odds` to constrain FPR too.
- **Aggregate representation matters** — use `demographic-parity`.
- **Well-calibrated risk scores** are required — use `calibration-
  parity`.

## Files

- `python/equal_opportunity.py` — from-scratch `per_group_tpr`,
  `equal_opportunity_diff/ratio`, and the Hardt post-processing
  `group_thresholds_for_tpr`. Demo on synthetic two-group data:
  single-threshold EOpp diff = 0.14 – 0.17; group-specific thresholds
  reduce it to ≤ 0.003 for target TPRs 0.60 / 0.75 / 0.90.
- `r/equal_opportunity.R` — `fairness` / `fairml` / `mlr3fairness`;
  `fairlearn.metrics.true_positive_rate_*` in Python.

## Assumptions & caveats

- **Only constrains TPR** — FPR gaps can widen after post-processing;
  audit both.
- **Group-specific thresholds** are legally / ethically controversial
  in some jurisdictions (US employment: prohibited by CRA-1991
  §106).
- **Target TPR** matters — very high or low targets constrain few
  thresholds and can approach randomness for small groups.
- **Requires ground truth Y** — must be trustworthy; measurement bias
  in Y contaminates the audit.
- **Impossibility** — with unequal base rates, EOpp + calibration parity
  cannot both hold in general.

## Related in this repo

- `equalized-odds` — stronger criterion, also constrains FPR.
- `demographic-parity` — different criterion, ignores Y.
- `calibration-parity` — sibling criterion; often conflicts with EOpp.
- `equalized-odds-postprocessing` — same-family mitigation for both
  TPR and FPR.
- `adversarial-debiasing`, `exponentiated-gradient-reduction` — in-
  training alternatives.

## Run

```
python techniques/equal-opportunity/python/equal_opportunity.py
Rscript techniques/equal-opportunity/r/equal_opportunity.R
```

**Refs:** Hardt, M., Price, E. & Srebro, N. "Equality of opportunity in supervised learning." *NeurIPS*, 2016; Chouldechova, A. "Fair prediction with disparate impact." *Big Data*, 2017.

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

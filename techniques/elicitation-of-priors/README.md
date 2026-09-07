# Elicitation of Priors (Reference §23.20)

Kadane et al. (1980); O'Hagan et al. (2006, *Uncertain Judgements*).
Structured methods for quantifying an expert's belief as a probability
distribution.

## Workflow

1. Ask the expert for a small number of quantitative summaries —
   **median**, **quartiles**, **tail probabilities**, plausible
   range.
2. Fit a parametric distribution whose implied summaries best match
   the elicited values.
3. Feed as **prior** into a Bayesian analysis; **pool** across
   experts (linear / logarithmic / SHELF).

## Files

- `python/elicitation_of_priors.py` — closed-form Normal fit +
  Nelder-Mead Beta fit from median + IQR + linear opinion pool.
  Demo: Expert A (median 0.40, IQR 0.20) → Beta(4.51, 6.65);
  Expert B (median 0.60, IQR 0.10) → Beta(26.31, 17.62); equal-
  weight linear pool mean 0.501. Normal example matches
  (0.10, 0.90) quantiles.
- `r/elicitation_of_priors.R` — `SHELF` (Oakley & O'Hagan),
  `rriskDistributions`, `bayestestR::describe_prior`, `prevalence`
  (R); `pyshelf` (unofficial), `scipy + optimize` (Python).

## When to use

- **Bayesian analysis with little data** — clinical trials at
  interim / early phase, novel exposures, small-area studies.
- **Regulatory submissions** — FDA / EMA increasingly accept
  Bayesian designs with elicited priors.
- **Model comparison via Bayes factors** — sensitivity to prior
  hinges on defensible elicitation.
- **Structured expert judgement** — safety, risk, health-tech
  assessment.

## When NOT to use

- **Large data, weak effects** — the likelihood dominates; a
  weakly-informative prior suffices.
- **Sensitive contexts where expert bias is likely** — consider
  supervised elicitation (SHELF) with training and adversarial
  probes.
- **When experts disagree wildly** — a linear pool blurs modes;
  present multiple priors and run robustness analyses.

## Assumptions & caveats

- **Anchoring / availability biases** — experts anchor on the first
  quantity elicited. Randomise order.
- **Overconfidence** — experts systematically give too-narrow
  intervals. Recalibrate with feedback questions.
- **Facilitator effects** — SHELF style requires a trained
  facilitator; document the protocol.
- **Pooling method** — linear pool preserves the mean but not
  necessarily the mode; logarithmic pool gives Kullback-averaged
  distributions.

## Related in this repo

- `bayesian-hierarchical-models`, `bayesian-glms`, `bayesian-linear-
  regression` — priors are consumed here.
- `conjugate-priors` — analytic forms elicitation targets.
- `posterior-predictive-checks`, `bayesian-model-comparison` —
  downstream sensitivity diagnostics.
- `bayesian-ab-testing` — a canonical use case.

## Run

```
python techniques/elicitation-of-priors/python/elicitation_of_priors.py
Rscript techniques/elicitation-of-priors/r/elicitation_of_priors.R
```

**Refs:** Kadane, J.B. et al. "Interactive elicitation of opinion for a normal linear model." *JASA*, 75(372): 845-854, 1980; O'Hagan, A. et al. *Uncertain Judgements: Eliciting Experts' Probabilities*, Wiley, 2006.

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

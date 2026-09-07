# Counterfactual Explanations (Reference §47.33)

Wachter, Mittelstadt & Russell (2017). Given a point `x` with an
adverse prediction, find the **smallest change** `x'` that flips the
prediction to the desired class:

    x' = argmin  d(x, x') + λ · L(f(x'), y_desired)

Common distances: L1 (encourages sparse changes), MAD-normalised L1
(Wachter), Gower for mixed types.

## Files

- `python/counterfactual_explanations.py` — Wachter-style gradient
  minimisation of `d(x, x') + λ (f(x') − y*)²` for a logistic
  classifier from scratch. Demo (4-feature loan approval, x*
  currently rejected p=0.143): counterfactual concentrates
  Δincome = +2.46, leaving other features almost untouched
  (sparse), post-prob = 0.77 (approved).
- `r/counterfactual_explanations.R` — `iml::Counterfactuals` (R);
  `alibi.explainers.CounterfactualProto`, DiCE, CARLA (Python).

## When to use

- **GDPR-style "right to explanation"** — actionable individual
  advice.
- **Loan / hiring / clinical decision** — "what would need to
  change for approval?"
- **Recourse research** — study whether recommendations are
  actionable / fair.

## When NOT to use

- **Immutable protected features** — must constrain `x'` to
  actionable subsets (age, sex not modifiable).
- **Very high-dim / structured inputs (images, text)** — needs
  perceptual distance / generative-model latents.
- **Population-level analysis** — counterfactuals are individual;
  use PDP / ALE for global.

## Assumptions & caveats

- **Distance choice** dominates the recommendation — L1 → sparse,
  L2 → distributed, Gower for mixed types.
- **Actionability** — the changes must be feasible in the real
  world; enforce constraints (age cannot decrease).
- **Data-manifold consistency** — the counterfactual should lie on
  or near the training data manifold; use prototypes (CounterfactualProto)
  or generative priors.
- **Diversity** — DiCE produces multiple different counterfactuals
  so the user can pick the acceptable one.

## Related in this repo

- `lime-local-explanations`, `anchor-explanations`,
  `shap-values`, `integrated-gradients`,
  `explainable-boosting-machine` — XAI siblings.
- `counterfactual-fairness` — related fairness cousin.
- `dpo-direct-preference-optimization` — different "counterfactual"
  concept (preferences).

## Run

```
python techniques/counterfactual-explanations/python/counterfactual_explanations.py
Rscript techniques/counterfactual-explanations/r/counterfactual_explanations.R
```

**Refs:** Wachter, S., Mittelstadt, B. & Russell, C. "Counterfactual explanations without opening the black box: automated decisions and the GDPR." *Harvard Journal of Law & Technology*, 31: 841-887, 2017; Mothilal, R.K., Sharma, A. & Tan, C. "Explaining machine learning classifiers through diverse counterfactual explanations." *FAT*, 2020.

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

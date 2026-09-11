# DiCE — Diverse Counterfactual Explanations (Reference §47.254)

Mothilal, Sharma & Tan (2020). Given a point `x` with
prediction `f(x)`, find a SET of counterfactuals `{c_1, ...,
c_k}` that flip the prediction while being:

- **proximal**: each `c_i` close to `x`
- **diverse**: `c_i`s far from each other
- **valid**: each `c_i` flips the prediction

Optimises `L = validity + λ_p · proximity − λ_d · diversity`.
More useful than a single counterfactual because it reveals
MULTIPLE valid recourse paths.

## Files

- `python/dice_diverse_counterfactuals.py` — random-search +
  local-move DiCE (simplified). Demo: 5-feature gradient-boost
  classifier on synthetic data. Started from a point with
  P(class=1) = 0.20; found 3 counterfactuals all with P ≥ 0.55
  through different feature changes, mean pairwise L1 diversity
  21.8 — three genuinely distinct recourse routes.
- `r/dice_diverse_counterfactuals.R` — `counterfactuals`
  package (WhatIf / NICE / MOC), `iml::Predictor`,
  `reticulate → dice-ml` (R); `dice-ml`,
  `alibi.CounterfactualProto`, from-scratch (Python).

## When to use

- **Model explanations for individuals** — "what could I change
  to get a loan?"
- **Actionable recourse** — offering users multiple valid
  paths, not just one.
- **Feature-importance-in-context** — DiCE surfaces which
  features CAN change to affect the outcome.

## When NOT to use

- **Structural / causal questions** — DiCE finds correlational
  counterfactuals; use causal counterfactual explanations for
  intervention semantics.
- **High-dimensional images** — DiCE-genetic or diffusion-based
  CFs are more appropriate.
- **Model is a black-box API with strict rate limits** — DiCE
  needs many forward passes; use surrogate-based methods.

## Assumptions & caveats

- **Feasibility of features** — DiCE does not know which
  features are actionable; supply an `immutable` list in real
  use (age, race, birthdate).
- **Plausibility** — CFs may land off the data manifold; use
  DiCE-ML's variational / genetic backends for realistic
  points.
- **Diversity-proximity trade-off** governed by `λ_p, λ_d` —
  tune per stakeholder needs.
- **Local search** in our demo is a stochastic heuristic; for
  smooth models use gradient-based DiCE.

## Related in this repo

- `counterfactual-explanations` — the single-CF baseline.
- `shapley-permutation-explainer`, `lime-local-explanations`,
  `anchor-explanations` — other local explainers.
- `individual-fairness` — related recourse framing.
- `counterfactual-fairness` — causal counterfactual cousin.

## Run

```
python techniques/dice-diverse-counterfactuals/python/dice_diverse_counterfactuals.py
Rscript techniques/dice-diverse-counterfactuals/r/dice_diverse_counterfactuals.R
```

**Refs:** Mothilal, R.K., Sharma, A. and Tan, C. "Explaining Machine Learning Classifiers Through Diverse Counterfactual Explanations." In *ACM FAT\**, pp. 607-617, 2020.

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

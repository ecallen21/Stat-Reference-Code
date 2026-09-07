# Anchor Explanations (Reference §47.32)

Ribeiro, Singh & Guestrin (2018, AAAI). Extends LIME with an
**IF-THEN rule** ("anchor") whose precision guarantees the model
gives the same prediction whenever the anchor's conditions hold,
with high probability:

    IF (X_j ∈ interval_j) AND (X_k = category_k) …
    THEN pred = c,  precision ≥ τ

Search: beam search over feature-value conditions; precision is
estimated by perturbing non-anchor features and checking model
stability.

## Files

- `python/anchor_explanations.py` — greedy anchor finder from
  scratch on a rule-based toy classifier `y = 1 iff x0 > 0.5 AND
  x2 == 0`. Demo: at x* = (0.8, 0.5, 0, 0.3) the greedy search
  correctly picks features 0 and 2 with precision 1.00 — matches
  the ground-truth logical rule.
- `r/anchor_explanations.R` — no R port; describes
  `alibi.explainers.AnchorTabular/Text/Image`, Ribeiro's `anchor`
  package (Python).

## When to use

- **Interpretable local explanation** with a **crisp rule** the user
  can verbalise ("IF income > 50k AND age > 30 THEN loan approved").
- **Model debugging** — find sufficient conditions triggering an
  output.
- **High-stakes contexts** — audit / regulatory where the precision
  guarantee is required.

## When NOT to use

- **Continuous smooth outputs** — anchors are for classification (or
  discretised regression).
- **Very high-dim images / text without segmentation** — anchors
  need discrete units (superpixels, words); costly for pixel-level.
- **Sparse features** — greedy search may fail to find high-
  precision rule with limited coverage.

## Assumptions & caveats

- **Precision estimator noise** — perturbation sample size trades
  runtime vs precision estimate variance.
- **Coverage** matters — a precise rule that fires rarely tells the
  user little.
- **Categorical / continuous mixing** — discretise continuous
  features into bins for anchor conditions.
- **Contrast with LIME** — LIME gives coefficients (local slopes);
  anchors give rules (local logic).

## Related in this repo

- `lime-local-explanations`, `shap-values`,
  `integrated-gradients`, `counterfactual-explanations` — XAI
  siblings.
- `decision-tree`, `explainable-boosting-machine` — inherently
  interpretable models.

## Run

```
python techniques/anchor-explanations/python/anchor_explanations.py
Rscript techniques/anchor-explanations/r/anchor_explanations.R
```

**Refs:** Ribeiro, M.T., Singh, S. & Guestrin, C. "Anchors: high-precision model-agnostic explanations." *AAAI*, 2018; Molnar, C. *Interpretable Machine Learning*, 2nd ed., 2022.

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

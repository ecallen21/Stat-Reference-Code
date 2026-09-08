# Feature-Attribution Stability (Reference §47.52)

Yeh et al 2019 'On the (in)fidelity and sensitivity of
explanations', NeurIPS; Alvarez-Melis & Jaakkola 2018 'On the
robustness of interpretability methods'. Quantifies how much a
feature-attribution method's output changes under:

    Local Lipschitz:   L(x) = max_{x' in B(x, ε)}  ‖φ(x) − φ(x')‖ / ‖x − x'‖
    Bootstrap Jaccard: J    = |topk(A) ∩ topk(B)| / |topk(A) ∪ topk(B)|

Two attribution methods can agree at a point yet be **unstable** —
tiny input perturbations or model retrains reshuffle top-k
features. A stability audit is mandatory before releasing an
XAI-based explanation.

## Files

- `python/attribution_stability.py` — local-Lipschitz numerical-
  gradient estimate + bootstrap top-k Jaccard. Demo (n=400,
  p=8, gradient boosting + permutation importance):
  - **independent features**: Jaccard = 1.00 across 10 bootstraps.
  - **correlated features** (5 near-copies of x₀-x₂): Jaccard
    drops to 0.43 ± 0.21 — attributions swap across resamples.
- `r/attribution_stability.R` — `iml::FeatureImp` + custom;
  `shap`, `captum`, `alibi`, from-scratch (Python).

## When to use

- **Before publishing / regulating** on XAI outputs.
- **Comparing attribution methods** — LIME vs SHAP vs Integrated
  Gradients vs permutation importance.
- **Correlated-feature diagnostics** — swap-across-resample is a
  red flag.
- **Trust / governance** — MODEL RISK MANAGEMENT frameworks (SR
  11-7) demand stability evidence.

## When NOT to use

- **Single explanations** for user-facing tooltips — stability is
  a MODEL-level property, not a per-instance one.
- **When the model itself is unstable** (very high-variance base
  learner) — mistake will propagate.

## Assumptions & caveats

- **ε and k** need domain choice; report sensitivity.
- **Attribution ties** — Jaccard ties inflate agreement; use
  Kendall τ over the full ranking as a companion metric.
- **Numerical gradients** are expensive; use auto-grad when
  possible.
- **Bootstrap variance ≠ true variance** — use nested-CV or
  parametric-bootstrap alternatives.
- **Faithfulness ≠ stability** — a stable but wrong attribution
  is still misleading; audit fidelity too.

## Related in this repo

- `shap-values`, `shap-interactions`, `lime-local-explanations`,
  `anchor-explanations`, `integrated-gradients`, `pdp-ice-plots`,
  `ale-accumulated-local-effects`, `counterfactual-explanations`,
  `explainable-boosting-machine`, `cate-clustering-ice` —
  attribution methods to audit.
- `bootstrap-optimism-correction`, `nonparametric-bootstrap`,
  `stability-selection` — bootstrap-based stability toolset.
- `permutation-tests`, `model-x-knockoffs` — significance testing
  cousins.
- `model-monitoring-metrics`, `model-cards` — governance / audit.

## Run

```
python techniques/attribution-stability/python/attribution_stability.py
Rscript techniques/attribution-stability/r/attribution_stability.R
```

**Refs:** Yeh, C.-K., Hsieh, C.-Y., Suggala, A.S., Inouye, D.I. & Ravikumar, P. "On the (in)fidelity and sensitivity of explanations." *NeurIPS*, 2019; Alvarez-Melis, D. & Jaakkola, T. "On the robustness of interpretability methods." *WHI ICML*, 2018.

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

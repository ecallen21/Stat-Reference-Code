# Integrated Gradients (Reference §47.30)

Sundararajan, Taly & Yan (2017, ICML). Attribution method for
differentiable models satisfying **sensitivity** and **implementation
invariance** axioms:

    IG_i(x) = (x_i − x_i^base) · ∫₀¹  ∂f(x^base + α (x − x^base)) / ∂x_i  dα

Completeness: `Σ IG_i(x) = f(x) − f(x^base)`.

## Files

- `python/integrated_gradients.py` — trapezoidal path-integral IG
  from scratch for a differentiable analytic function. Demo
  (`f = tanh(x0) + 0.5 x1² + 0.2 x0·x2`, x=(1,2,3), baseline=0):
  IG = (1.062, 2.000, 0.300); sum = 3.362 = f(x) − f(0)
  exactly (completeness ✓). "Gradient × input" gives (1.02, 4.0,
  0.6) → sums to 5.6, fails completeness.
- `r/integrated_gradients.R` — torch (Posit) + custom, `innsight`
  (R, in-dev); `captum.attr.IntegratedGradients`,
  `alibi.explainers.IntegratedGradients`, from-scratch (Python).

## When to use

- **Explain a differentiable model's prediction** — neural
  networks (CNN, transformer, MLP).
- **Regulatory / medical model attribution** — completeness gives
  auditable per-feature contribution.
- **Gradient saliency inadequate** — saturating activations break
  saliency; IG doesn't.

## When NOT to use

- **Non-differentiable models** — random forests, boosting; use
  SHAP or LIME.
- **Extremely high-dim inputs** with expensive gradients (huge
  transformers) — approximate with fewer steps.
- **Comparative importance across inputs** — IG values depend on
  baseline; changing baseline shifts attribution.

## Assumptions & caveats

- **Baseline choice** dominates the story — zero / all-black /
  Gaussian noise / class-mean can each yield different pictures.
- **Path integral discretisation** — 20–100 steps usually suffice;
  check convergence by doubling.
- **Gradient noise / adversarial fragility** — SmoothGrad-IG
  averages IG over Gaussian-perturbed x for stability.
- **Attribution vs causation** — IG explains sensitivity of the
  model, not real causation in the world.

## Related in this repo

- `lime-local-explanations`, `shap-values`,
  `explainable-boosting-machine` — explanation cousins.
- `jacobian-regularization`, `deep-mlp-backprop`,
  `attention-mechanism` — differentiable-model machinery.

## Run

```
python techniques/integrated-gradients/python/integrated_gradients.py
Rscript techniques/integrated-gradients/r/integrated_gradients.R
```

**Refs:** Sundararajan, M., Taly, A. & Yan, Q. "Axiomatic attribution for deep networks." *ICML*, 2017; Kokhlikyan, N. et al. *Captum: A unified and generic model interpretability library for PyTorch*, 2020.

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

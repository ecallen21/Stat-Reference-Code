# Permutation / Kernel SHAP Explainer (Reference §47.130)

Štrumbelj & Kononenko (2010); Lundberg & Lee (2017). Permutation
SHAP averages the marginal contribution of feature i to f as it is
INSERTED at various orderings; Kernel SHAP fits a Shapley-weighted
linear regression on random subset masks. Both estimate the same
Shapley values in expectation.

## Files

- `python/shapley_permutation_explainer.py` — from-scratch
  permutation SHAP (all d! orderings for small d, MC otherwise) +
  Kernel SHAP with the correct Shapley kernel. Demo (d=4, f =
  x₀ + 2x₁ − x₂ + 0.5 x₁x₃, x=(1,1,1,1) vs baseline zero, target
  Σφ = 2.5):
  - perm-exact (24 perms):    φ = (+1.00, +2.25, −1.00, +0.25)
  - perm-MC (200 perms):      φ = (+1.00, +2.24, −1.00, +0.26)
  - Kernel SHAP (800 samples): φ = (+1.00, +2.26, −1.01, +0.24)
  All three sum to **+2.500** (efficiency).
- `r/shapley_permutation_explainer.R` — `iml::Shapley`,
  `fastshap`, `DALEX::variable_attribution` (R); `shap.KernelExplainer`,
  `shap.PermutationExplainer` (Python).

## When to use

- **Any black-box explanation** — model-agnostic.
- **Feature attributions with efficiency, symmetry, additivity
  guarantees** — Shapley axioms.
- **When a tree-fast alternative is unavailable** — Kernel SHAP is
  the fallback.

## When NOT to use

- **Tree ensembles** — TreeSHAP is polynomial-time; use it.
- **Real-time explanations** — Kernel SHAP is O(samples × f-calls),
  can be slow.
- **Very high d** — exact enumeration exponential; use Owen sampling
  or grouped features.

## Assumptions & caveats

- **Baseline** matters — mean / median / all-zero / conditional
  drops give different attributions.
- **Interventional vs conditional** SHAP — different assumptions on
  correlated features.
- **Sample size** for Kernel SHAP — variance falls with more subsets.
- **Additivity** across background samples for stability; average
  multiple baselines.

## Related in this repo

- `shap-values`, `shap-interactions`, `lime-local-explanations`,
  `anchor-explanations`, `integrated-gradients`,
  `counterfactual-explanations`, `pdp-ice-plots`,
  `ale-accumulated-local-effects`, `grad-cam-saliency`,
  `smoothgrad-saliency`, `attribution-stability` — XAI stack.
- `friedmans-h-statistic`, `cate-clustering-ice` — interaction
  cousins.
- `permutation-tests` — same sampling idea for hypothesis testing.

## Run

```
python techniques/shapley-permutation-explainer/python/shapley_permutation_explainer.py
Rscript techniques/shapley-permutation-explainer/r/shapley_permutation_explainer.R
```

**Refs:** Štrumbelj, E. & Kononenko, I. "An efficient explanation of individual classifications using game theory." *JMLR* 11: 1-18, 2010; Lundberg, S.M. & Lee, S.-I. "A unified approach to interpreting model predictions." *NeurIPS*, 2017.

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

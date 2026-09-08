# SHAP Interaction Values (Reference §47.40)

Lundberg, Erion & Lee (2018, arXiv:1802.03888). Extension of SHAP
that decomposes each prediction into a **matrix** of pairwise
attributions:

    φ_i(x) = φ_ii(x) + Σ_{j ≠ i} φ_ij(x)

- `φ_ii` — main-effect contribution.
- `φ_ij` — pure interaction between features `i, j`.

Additive-only models: `φ_ij = 0` for `i ≠ j`.

## Files

- `python/shap_interactions.py` — exact Shapley + Shapley-
  interaction values by subset enumeration from scratch (small
  feature count). Demo (`f = x0 + x1 + 2 x0·x2`, x=(1,1,1,1),
  baseline 0): Shapley values (2, 1, 1, 0); interaction matrix
  correctly puts +1 on both (0, 2) and (2, 0), zero elsewhere.
- `r/shap_interactions.R` — `shapviz::sv_interaction`, `kernelshap`
  (R); `shap.TreeExplainer.shap_interaction_values`, from-scratch
  (Python).

## When to use

- **Diagnosing interactions in tree ensembles** — XGBoost, LGBM,
  CatBoost natively support `shap_interaction_values`.
- **Explaining pair-wise feature effects** — beyond marginal
  SHAP.
- **Validating additive assumptions** — near-zero off-diagonal
  supports additive interpretation.

## When NOT to use

- **Very high `d`** — cost `O(2^d)` in general; even tree-based
  fast algorithm is `O(TLD²)`.
- **Deep nets** — kernel-SHAP interaction estimates are noisy;
  use gradient-based attribution instead.
- **Interpretation of >2-way interactions** — Shapley-Taylor /
  n-way generalisations exist but harder to visualise.

## Assumptions & caveats

- **Baseline** matters — SHAP values are contributions vs a
  reference; changing baseline changes attribution.
- **Correlated features** — interventional vs conditional SHAP
  gives different answers.
- **Tree-fast algorithm** exact for tree ensembles; kernel-SHAP is
  approximate for other models.
- **Visualisation** — matrix heat-map or SHAP-force plots pair-
  wise; hard to read at large `d`.

## Related in this repo

- `shap-values` — the marginal SHAP baseline.
- `lime-local-explanations`, `anchor-explanations`,
  `integrated-gradients`, `counterfactual-explanations`,
  `pdp-ice-plots`, `ale-accumulated-local-effects`,
  `explainable-boosting-machine` — XAI cousins.
- `gradient-boosting`, `random-forest` — model families natively
  supporting fast interaction values.

## Run

```
python techniques/shap-interactions/python/shap_interactions.py
Rscript techniques/shap-interactions/r/shap_interactions.R
```

**Refs:** Lundberg, S.M., Erion, G.G. & Lee, S.-I. "Consistent individualized feature attribution for tree ensembles." *arXiv:1802.03888*, 2018; Shapley, L.S. "A value for n-person games." *Contributions to the Theory of Games*, 1953.

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

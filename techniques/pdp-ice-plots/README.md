# Partial Dependence + ICE Plots (Reference §47.34)

Friedman (2001, PDP); Goldstein et al. (2015, ICE). Global /
per-instance visualisation of a feature's marginal effect on a
black-box prediction.

## Definitions

    PD_j(v)         = (1/n) Σᵢ f(xᵢ with x_ij ← v)
    ICE_j^{(i)}(v)  = f(xᵢ with x_ij ← v)

PDP is the average of ICE curves. **Fanning ICE curves reveal
interactions** the PDP averages away.

**Centred ICE** (c-ICE) subtracts `f(xᵢ at v_min)` from each curve
so they start at 0 → compare shapes.

## Files

- `python/pdp_ice_plots.py` — from-scratch PDP + ICE computation.
  Demo (`f = 0.5·x0·x1 + 0.3·x2`, sole interaction on x0): PDP for
  x0 is ≈ 0 across the grid (averages away), while ICE curves for
  low / mid / high x1 quantiles fan out from slope −0.6 to +0.7 —
  the interaction is visible in ICE.
- `r/pdp_ice_plots.R` — `pdp::partial`, `ICEbox`,
  `iml::FeatureEffect`, `DALEX::model_profile` (R);
  `sklearn.inspection.partial_dependence`, `PDPbox` (Python).

## When to use

- **Global feature-effect visualisation** — how does the model
  respond to `x_j`?
- **Interaction detection** — fanning ICE curves.
- **Communicate a black-box model** to non-technical audiences.
- **Model debugging** — surprising PDP shapes flag training issues.

## When NOT to use

- **Correlated features** — PDP extrapolates into low-density
  regions; use ALE instead.
- **Very high-dim inputs** — PDP over pixels / tokens is
  uninteresting; use IG / SHAP.
- **Categorical outputs with many classes** — plot per-class PDP
  separately.

## Assumptions & caveats

- **Independence between `x_j` and other features** — PDP under
  correlation gives misleading extrapolation.
- **Grid choice** — use quantiles of the observed x_j, not equal
  spacing, to avoid extrapolation.
- **ICE alone** can be visually noisy; centre / colour by an
  interacting feature to spot structure.
- **Two-way PDP** shows interactions but harder to read at scale.

## Related in this repo

- `ale-accumulated-local-effects` — extrapolation-safe alternative
  to PDP.
- `shap-values`, `lime-local-explanations`,
  `anchor-explanations`, `integrated-gradients`,
  `counterfactual-explanations` — XAI cousins.
- `feature-importance` — global scalar summary.

## Run

```
python techniques/pdp-ice-plots/python/pdp_ice_plots.py
Rscript techniques/pdp-ice-plots/r/pdp_ice_plots.R
```

**Refs:** Friedman, J.H. "Greedy function approximation: a gradient boosting machine." *Annals of Statistics*, 29(5): 1189-1232, 2001; Goldstein, A. et al. "Peeking inside the black box: visualizing statistical learning with plots of individual conditional expectation." *JCGS*, 24(1): 44-65, 2015.

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

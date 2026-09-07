# ALE -- Accumulated Local Effects (Reference §47.35)

Apley & Zhu (2020, JRSS-B). Fixes PDP's extrapolation problem when
features are correlated: integrate LOCAL differences within thin
bins of `x_j` and accumulate.

## Algorithm

1. Bin `x_j` into `K` quantile intervals.
2. For each bin `k`, average
   `f(x with x_j = upper) − f(x with x_j = lower)` over
   observations whose `x_j` falls in bin `k`.
3. Accumulate (cumsum) across bins.
4. Centre so `ALE(v_mean) = 0`.

Unlike PDP, ALE never asks the model about combinations of `x_j` and
other features that do not occur in the data.

## Files

- `python/ale_accumulated_local_effects.py` — 1-D ALE from
  scratch. Demo (n=400, x0 & x1 correlated ρ=0.9, `f = tanh(x0) +
  0.3 x1`): ALE curve matches `tanh(x0) − mean(tanh)` at every
  grid point to ~0.05 — the true nonlinear shape is recovered
  without extrapolation.
- `r/ale_accumulated_local_effects.R` — `ALEPlot`,
  `iml::FeatureEffect(method='ale')`, `DALEX::model_profile`
  (R); `alibi.explainers.ALE`, `PDPbox`, from-scratch (Python).

## When to use

- **Correlated features** — PDP extrapolates unseen combinations;
  ALE stays inside the data.
- **Global feature effect** — same use case as PDP but safer.
- **Two-way ALE** — replaces two-way PDP for interaction visualisation.

## When NOT to use

- **Independent features** — PDP and ALE agree; PDP is simpler.
- **Very small n** — bin counts thin out; use fewer bins or a
  smoothed variant.
- **Discrete / categorical `x_j`** — apply category-specific ALE
  (Apley §5).

## Assumptions & caveats

- **Bin edges** — quantile bins avoid tail sparsity; too few bins
  smooth the effect, too many are noisy.
- **Centring** — subtract weighted mean over the observed density.
- **Uncertainty** — ALE has no built-in CI; bootstrap or
  cross-validation for stability.
- **Second-order ALE** requires two-feature bin grid; expensive
  for large n and fine bins.

## Related in this repo

- `pdp-ice-plots` — the extrapolation-prone alternative.
- `shap-values`, `lime-local-explanations`,
  `anchor-explanations`, `integrated-gradients`,
  `counterfactual-explanations` — XAI cousins.
- `feature-importance` — global scalar importance.

## Run

```
python techniques/ale-accumulated-local-effects/python/ale_accumulated_local_effects.py
Rscript techniques/ale-accumulated-local-effects/r/ale_accumulated_local_effects.R
```

**Refs:** Apley, D.W. & Zhu, J. "Visualizing the effects of predictor variables in black box supervised learning models." *JRSS-B*, 82(4): 1059-1086, 2020; Molnar, C. *Interpretable Machine Learning*, 2nd ed., ch 5.3, 2022.

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

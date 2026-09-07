# LIME -- Local Interpretable Model-Agnostic Explanations (Reference §47.29)

Ribeiro, Singh & Guestrin (2016, KDD). Explain a black-box model
prediction at point `x*` by fitting a **local sparse linear
surrogate**:

1. Sample perturbed inputs `z_i` around `x*`.
2. Get black-box prediction `f(z_i)`.
3. Weight `z_i` by proximity `π(z_i, x*)`.
4. Fit a sparse linear model (LASSO) on `(z_i, f(z_i))` with
   weights.

Coefficients are the **local feature importances** for `x*`.

## Files

- `python/lime_local_explanations.py` — tabular LIME with Gaussian
  perturbation + kernel weighting + ridge-approx LASSO from scratch.
  Demo (black box `exp(x0) − 2 sin(3 x1) + x2³`, x* = (0.5, 0, 1)):
  LIME coefs (+1.73, −4.34, +3.33) match analytic gradients (+1.65,
  −6.0, +3.0) — LIME's local slope is close to the true derivative.
- `r/lime_local_explanations.R` — `lime`, `DALEX + iBreakDown`,
  `vip / pdp / iml` (R); `lime`, `alibi` (Python).

## When to use

- **Black-box model explanation** — random forests, boosted
  trees, deep nets when you need one-prediction attribution.
- **Model debugging** — spot features driving surprising
  predictions.
- **Regulatory transparency** — "why did the model recommend X?"

## When NOT to use

- **Model is already interpretable** — coefficients from
  regression / GAM suffice.
- **Global insights needed** — LIME is local; use permutation
  importance / SHAP for global.
- **Stability critical** — LIME can be unstable across
  perturbation seeds; use SHAP or Anchors.

## Assumptions & caveats

- **Perturbation scheme** matters — Gaussian for continuous,
  category-flip for categorical, superpixel-mask for images.
- **Kernel width** trades locality vs SNR; too narrow → noise,
  too wide → global slope.
- **Local linearity** — LIME is a first-order approximation;
  strong non-linearity at `x*` biases explanation.
- **Instability across seeds** — average over multiple runs or
  use SHAP.

## Related in this repo

- `shap-values` — cousin attribution method (game-theoretic).
- `integrated-gradients` — gradient-based attribution for
  differentiable models.
- `explainable-boosting-machine` — inherently interpretable
  boosted GAM.
- `counterfactual-fairness` — related counterfactual-explanation
  cousin.

## Run

```
python techniques/lime-local-explanations/python/lime_local_explanations.py
Rscript techniques/lime-local-explanations/r/lime_local_explanations.R
```

**Refs:** Ribeiro, M.T., Singh, S. & Guestrin, C. "Why should I trust you?: explaining the predictions of any classifier." *KDD*, 2016; Molnar, C. *Interpretable Machine Learning*, 2nd ed., 2022.

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

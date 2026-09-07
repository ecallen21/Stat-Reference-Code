# Explainable Boosting Machine (EBM) (Reference §47.31)

Nori, Jenkins, Koch & Caruana (2019); Lou et al. (2013). An
**inherently interpretable** gradient-boosted GAM:

    f(x) = β₀ + Σⱼ fⱼ(xⱼ) + Σ_{j<k} f_{jk}(xⱼ, xₖ)

Each `fⱼ` is fit by a tiny bag of decision-tree stumps, round-robin
over features. The per-feature shape function is directly plottable
and interpretable.

## Files

- `python/explainable_boosting_machine.py` — round-robin stump
  boosting on residuals from scratch. Demo (n=800, p=4,
  `y = 1.5 tanh(x0) + 0.5 x1² + 0.3 x2 + noise`): R² = 0.916; EBM
  shape `f_0(x0)` closely tracks `1.5·tanh(x0)`; irrelevant `x3`
  gets max magnitude 0.15.
- `r/explainable_boosting_machine.R` — `interpret` (Microsoft R
  wrapper), `mboost`, `gbm + shapley` (R); `interpret / interpret-
  community`, `pyGAM`, `HistGradientBoosting` (Python).

## When to use

- **You need accuracy AND interpretability** — audit, credit,
  clinical prediction.
- **Feature-level effect shapes** — clinician / stakeholder wants
  a curve, not a coefficient.
- **Regulatory / GDPR / FDA settings** — inherent explanations
  avoid post-hoc SHAP / LIME instability.

## When NOT to use

- **Tabular data with heavy interactions** — EBM's pairwise-only
  interactions may miss higher-order structure; XGBoost / LightGBM
  may edge past.
- **Images / text** — EBM is tabular; use CNNs / transformers with
  post-hoc IG / SHAP.
- **Very small n** — the round-robin stumps overfit; use GAM /
  spline models.

## Assumptions & caveats

- **Additive structure** — pairwise interactions optional but
  bounded; not suitable for high-order interactions.
- **Bagging + cyclic order** matters — stochastic; multiple runs
  give slightly different shapes.
- **Categorical handling** — one-hot or bin encoding; EBM's
  official implementation handles both.
- **Global vs local** — EBM gives GLOBAL shape functions;
  LIME / SHAP give LOCAL attributions.

## Related in this repo

- `gam`, `gamlss`, `additive-quantile-regression` — statistical
  cousins.
- `gradient-boosting`, `random-forest` — boosting siblings.
- `lime-local-explanations`, `shap-values`,
  `integrated-gradients` — post-hoc explanation methods.

## Run

```
python techniques/explainable-boosting-machine/python/explainable_boosting_machine.py
Rscript techniques/explainable-boosting-machine/r/explainable_boosting_machine.R
```

**Refs:** Nori, H. et al. "InterpretML: A unified framework for machine learning interpretability." *arXiv:1909.09223*, 2019; Lou, Y. et al. "Accurate intelligible models with pairwise interactions." *KDD*, 2013; Hastie, T.J. & Tibshirani, R. *Generalized Additive Models*, Chapman & Hall, 1990.

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

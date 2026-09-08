# Friedman's H-Statistic (Reference §47.95)

Friedman & Popescu (2008). Model-agnostic interaction detection.
For a fitted black-box f and features j, k:

    H²_{jk} = Σᵢ [PD_{jk}(x_ij, x_ik) − PD_j(x_ij) − PD_k(x_ik)]²
              / Σᵢ PD_{jk}(x_ij, x_ik)²

where PD_j is the partial-dependence function. H² ∈ [0, 1]:
0 = purely additive, 1 = pure interaction. Companion to SHAP-
interactions but works with ANY predictor.

## Files

- `python/friedmans_h_statistic.py` — from-scratch H² with
  Monte-Carlo partial-dependence + `sklearn` gradient boosting.
  Demo (n=400, 4 features):
  - **Additive truth** `2·x₀ + x₁² − 0.5·x₂`: all H(j, k) ≤ 0.07.
  - **With (0, 2) interaction** `+ 3·x₀·x₂`: H(0, 2) = 0.83, all
    others ≤ 0.20 (correctly identifies the true pair).
- `r/friedmans_h_statistic.R` — `iml::Interaction`,
  `pre` (R); `sklearn.inspection` + custom, from-scratch (Python).

## When to use

- **Detecting pairwise interactions** in tree ensembles / any
  regressor.
- **Diagnosing model complexity** — near-zero H² justifies
  additive interpretation.
- **Feature-engineering guidance** — flag pairs to include
  explicitly as interaction terms.
- **Complement SHAP interactions** — model-agnostic vs
  tree-specific.

## When NOT to use

- **Very high d** — O(d²) pairs; combine with a shortlist from
  domain / SHAP.
- **Highly correlated features** — PD extrapolates outside the
  data manifold; use ALE-based interaction (Apley-Zhu 2020).
- **Small n** — H estimator has high variance; bootstrap CIs
  essential.

## Assumptions & caveats

- **PD marginalisation** — assumes feature independence; use
  ALE or conditional PD for correlated features.
- **H² sensitivity** to grid resolution — Monte-Carlo integration
  reduces this but is slower.
- **Interpretation** — H² is a RELATIVE share of interaction,
  not an absolute effect size; combine with plots.
- **Higher-order** — Friedman-Popescu extend to 3-way interactions;
  numerically noisier.

## Related in this repo

- `shap-values`, `shap-interactions`, `pdp-ice-plots`,
  `ale-accumulated-local-effects`, `cate-clustering-ice`,
  `attribution-stability` — XAI toolkit.
- `gradient-boosting`, `random-forest`,
  `bart-bayesian-additive-regression-trees`,
  `explainable-boosting-machine`,
  `catboost-ordered-boosting` — model families.
- `interaction-terms`, `varying-coefficient-model`,
  `mixed-logit-mnl` — model-based interaction cousins.

## Run

```
python techniques/friedmans-h-statistic/python/friedmans_h_statistic.py
Rscript techniques/friedmans-h-statistic/r/friedmans_h_statistic.R
```

**Refs:** Friedman, J.H. & Popescu, B.E. "Predictive learning via rule ensembles." *Ann Appl Stat* 2(3): 916-954, 2008.

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

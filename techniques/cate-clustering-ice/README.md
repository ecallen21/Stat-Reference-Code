# CATE via ICE-Curve Clustering (Reference §47.51)

Zhao & Hastie (2019/2020). ICE (Individual Conditional
Expectation) plots (Goldstein-Kapelner-Bleich-Pitkin 2015) show
predicted outcome as a function of a treatment / covariate for
each unit separately. Clustering the shapes reveals **subgroups**
with different counterfactual response profiles — an interpretable
summary of the Conditional Average Treatment Effect (CATE).

Recipe: (1) fit any regression, (2) for each subject build
`y_i(t)` at a grid of `t`, (3) K-means the centered curves,
(4) read subgroup mean shapes.

## Files

- `python/cate_clustering_ice.py` — ICE curve generation +
  K-means on centered profiles. Demo (n=800 with true moderator
  `x₂ ∈ {0,1}` that flips the sign of the treatment quadratic):
  cluster split by `x₂` — ARI(cluster, x₂) = 1.000. Cluster 0
  members are 100 % `x₂=1`; cluster 1 members are 0 % `x₂=1`.
- `r/cate_clustering_ice.R` — `ICEbox`, `iml::FeatureEffect`
  (R); `PyCEbox`, `dalex`, from-scratch (Python).

## When to use

- **Detecting HTE without pre-specified subgroups** — data-driven
  subgroup discovery.
- **Sensitivity / robustness checks** — do subgroups persist under
  bootstraps of the base model?
- **Explaining CATE-forest / X-learner output** — cluster the
  per-unit CATE curves to summarise.
- **Regulator-facing summaries** — interpretable subgroup
  descriptions.

## When NOT to use

- **Very small n** — clusters overfit; use domain-defined
  subgroups.
- **Highly correlated covariates** — ICE curves marginal over
  fixed x_j may misrepresent effects; use ALE + clustering.
- **Model is misspecified** — clusters reflect model quirks, not
  true CATE structure.
- **Time-varying treatments** — need dynamic-CATE clustering.

## Assumptions & caveats

- **PDP/ICE marginalisation** — extrapolates outside observed
  support of held-fixed covariates.
- **K choice** — silhouette, gap statistic, or Bayesian selection;
  small K for interpretability.
- **Centering** — cluster on `curve − curve.mean(axis=1)` to focus
  on shape, not baseline level.
- **Stability** — re-cluster on bootstrap resamples; report Jaccard
  of cluster assignments.

## Related in this repo

- `pdp-ice-plots` — the base ICE-curve tool.
- `ale-accumulated-local-effects` — correlated-feature alternative.
- `hte-uplift`, `causal-forest`, `bart-bayesian-additive-regression-
  trees`, `x-learner-t-learner` — CATE estimators to cluster.
- `shap-values`, `shap-interactions`, `lime-local-explanations`,
  `explainable-boosting-machine` — XAI complements.

## Run

```
python techniques/cate-clustering-ice/python/cate_clustering_ice.py
Rscript techniques/cate-clustering-ice/r/cate_clustering_ice.R
```

**Refs:** Goldstein, A., Kapelner, A., Bleich, J. & Pitkin, E. "Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation." *JCGS* 24(1): 44-65, 2015; Zhao, Q. & Hastie, T. "Causal interpretations of black-box models." *JBES* 39(1): 272-281, 2021.

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

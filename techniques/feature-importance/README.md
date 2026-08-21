# Feature Importance + Partial Dependence + ICE (Reference §26.16)

## Permutation importance (Breiman 2001; Fisher-Rudin-Dominici 2019)

```
baseline = model score on validation
for each feature j:
    shuffle column j on the validation set  (breaks x_j <-> y)
    drop = baseline - new score
    (repeat several shuffles; average)
```

Model-agnostic. More reliable than the tree-only **mean decrease in impurity** (MDI), which is biased toward high-cardinality features.

## Partial Dependence (PDP)

```
PDP_j(v) = E_{X_{−j}}[ f̂(x_j = v, X_{−j}) ]
```

Global average effect of `x_j` on the prediction, marginalizing over other features. Visualizes the relationship the model learned.

## Individual Conditional Expectation (ICE)

Per-observation curves — reveals **heterogeneous** effects that PDP averages away. If ICE lines are parallel → additive; if they cross → interactions.

## Files

- `python/feature_importance.py` — from-scratch permutation importance + PDP + ICE using the `decision-tree` module for a demo model. Demo on `y = 2x0 + x0·x1 + noise`, 3 noise features: permutation importance x0 = 1.90, x1 = 0.41, x2–x4 ≈ 0 (correctly identifies signal vs noise).
- `r/feature_importance.R` — pointers to `iml::FeatureImp / FeatureEffect`, `DALEX`, and `vip / pdp` packages.

## When to use

- **Interpretability of any black-box model** — trees, RF, GBM, neural nets, deep learning.
- **Model selection / feature-removal decisions** — drop features with permutation importance ~ 0.
- **Sanity checking** — before-and-after model diagnostics.

## Caveats

- **Correlated features** — permuting one creates unrealistic data; drop the correlated feature or use **grouped** permutation.
- **PDP hides interactions** — always look at ICE too when the model has interaction terms.
- **SHAP** (Lundberg-Lee 2017) — game-theoretic per-observation contributions; more expensive but locally faithful.

## Run

```
python techniques/feature-importance/python/feature_importance.py
Rscript techniques/feature-importance/r/feature_importance.R
```

**Refs:** Breiman, L. "Random forests." *Mach. Learn.* 45(1), 5–32, 2001; Friedman, J.H. "Greedy function approximation: a gradient boosting machine." *Ann. Stat.* 29(5), 1189–1232, 2001 (PDP); Goldstein, A. et al. "Peeking inside the black box: visualizing statistical learning with plots of individual conditional expectation." *J. Comp. Graph. Stat.* 24(1), 44–65, 2015 (ICE); Fisher, A., Rudin, C. & Dominici, F. "All models are wrong, but many are useful: learning a variable's importance by studying an entire class of prediction models simultaneously." *JMLR* 20(177), 1–81, 2019.

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

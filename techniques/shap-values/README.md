# SHAP Values (Reference §21.x extra)

**Sh**apley **A**dditive ex**P**lanations (Lundberg-Lee 2017) — the unique
allocation of the difference `f(x) − E[f(X)]` across features that satisfies:

- **Efficiency**: `Σ_i φ_i(x) = f(x) − E[f(X)]`.
- **Symmetry**: features with identical marginal contributions get equal shares.
- **Dummy**: a feature that never changes the prediction gets 0.
- **Additivity**: SHAP of an ensemble = sum of SHAP of components.

## Formula

```
φ_i(x) = Σ_{S ⊆ N \ {i}}  |S|! (n − |S| − 1)! / n!  ·  [ f(x_{S ∪ {i}}) − f(x_S) ]
```

`f(x_S)` is the expected model output when only the features in `S` are known;
missing features are integrated out against a background distribution
(row sample of the training data).

## Fast approximations

Exact enumeration is `O(2^p)` — infeasible past ~15 features. Three practical routes:

| Method | Complexity | Use when |
|---|---|---|
| **Kernel SHAP** (`shap.KernelExplainer`) | `O(n_sample · 2^p)` sampled | model-agnostic; small to medium `p` |
| **Tree SHAP** (`shap.TreeExplainer`) | `O(T L D²)` exact for trees | XGBoost / LightGBM / RF / CatBoost |
| **Deep SHAP** / DeepLIFT | forward-backward pass | neural networks |

## When to use

- **Local explanation** of one prediction — "why did the model score this row 0.83?"
- **Global feature importance** — mean `|φ_i|` across a validation set (better than tree-based split-gain importance).
- **Fairness / bias auditing** — decompose predictions across protected groups.
- **Model debugging** — high SHAP for a leaky feature or an unexpected interaction is a red flag.

## Files

- `python/shap_values.py` — exact enumeration of Shapley values for `p ≤ 12`; cross-check on a linear model (SHAP must equal `β_i · (x_i − E[x_i])` exactly). Demo (n=300, p=4 linear model): SHAP matches analytical formula to 1e-16; efficiency `Σ φ = 1.5370 = f(x) − E[f]` exactly.
- `r/shap_values.R` — `kernelshap`, `treeshap`, `fastshap`.

## Assumptions & caveats

- **Feature independence** is assumed by the marginal-expectation formulation. Correlated features lead to *causal-shapley* / *interventional* debates; see Aas et al. (2021) for conditional-vs-marginal SHAP.
- **Background matters** — different backgrounds (train mean, cluster medoids, a control cohort) give different SHAP values.
- **Additive on the log-odds / margin scale**, not on probability — for binary classifiers, sum SHAP on the logit and transform after.
- **Global importance from mean(|SHAP|)** differs from permutation importance; report both.
- **Interaction values** — `shap.TreeExplainer(interaction=True)` decomposes further into main effects + pairwise interactions.

## Run

```
python techniques/shap-values/python/shap_values.py
Rscript techniques/shap-values/r/shap_values.R
```

**Refs:** Shapley, L.S. "A value for n-person games." In *Contributions to the Theory of Games II*, Princeton UP, 1953; Lundberg, S.M. & Lee, S.-I. "A unified approach to interpreting model predictions." *NeurIPS*, 2017; Lundberg, S.M. et al. "From local explanations to global understanding with explainable AI for trees." *Nature Machine Intelligence* 2(1), 56–67, 2020.

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

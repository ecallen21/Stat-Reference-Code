# Last-Layer Bayesian / Neural Linear (Reference Ch 29 UQ)

Train the neural network **normally** (MLE), freeze all but the final
layer, then do **exact Bayesian linear regression** on the frozen
penultimate features. Also called *Neural Linear* (Snoek 2015) or
*Bayesian Last Layer (BLL)*; Kristiadi, Hein & Hennig (2020) proved that
even this partial Bayesian treatment fixes ReLU-network overconfidence
far from data.

## Formula

Let `φ(x)` be the frozen penultimate feature map. Prior + likelihood:

```
w    ~ N(0, σ_w² I)
y | x, w ~ N(φ(x)ᵀ w, σ_n²)
```

Posterior (Gaussian, closed form):

```
Σ_w = (Φᵀ Φ / σ_n² + I / σ_w²)⁻¹
μ_w =  Σ_w · Φᵀ y / σ_n²
```

Predictive:

```
μ*(x) = φ(x)ᵀ μ_w
σ*²(x) = φ(x)ᵀ Σ_w φ(x) + σ_n²
```

The `+ σ_n²` term is aleatoric noise; the quadratic form is the epistemic
piece, which grows in directions of the feature space poorly covered by
training data.

## When to use

- **Small extra cost on top of an existing trained network** — just do
  a linear solve.
- **Post-hoc calibration** without retraining.
- **Bayesian optimisation / active learning** where you need a
  well-behaved acquisition function.
- **Regression and classification alike** — for classification use
  Laplace at the last layer (`laplace-torch`).

## When NOT to use

- **Features are the bottleneck** — if the penultimate layer collapses
  OOD inputs onto in-distribution features, BLL cannot recover epistemic
  uncertainty. Combine with distance-aware layers (DUE, SNGP).
- **You need full posterior over all weights** — see `bayesian-neural-
  network` / `swag`.

## Files

- `python/last_layer_bayesian.py` — MLP trained via SGD; features frozen;
  exact `N(μ_w, Σ_w)` posterior over the last layer; predictive mean +
  variance closed form on a test grid.
- `r/last_layer_bayesian.R` — `reticulate` + `laplace-torch` (Python) or
  `sklearn.linear_model.BayesianRidge` on exported features; `brms` /
  `rstanarm` for a fully Bayesian last layer in R.

## Assumptions & caveats

- **Feature map is fixed** — errors in `φ(·)` are ignored; the posterior
  under-represents structural uncertainty.
- **σ_w, σ_n** are hyperparameters — set by marginal-likelihood
  maximisation (empirical Bayes) or held out.
- **Distance-aware features** (SNGP, DUE) massively improve BLL on OOD;
  vanilla ReLU features can still be overconfident.
- **Classification** — use the Laplace / probit approximation for the
  softmax marginal, `laplace-torch` covers this.

## Related in this repo

- `bayesian-linear-regression` — the closed-form solver used on top.
- `laplace-approximation` — Laplace over the full network (existing).
- `deep-ensembles`, `mc-dropout`, `swag`, `bayesian-neural-network` —
  the sibling UQ methods.
- `gaussian-process-regression` — the "infinite-neuron" limit of BLL.

## Run

```
python techniques/last-layer-bayesian/python/last_layer_bayesian.py
Rscript techniques/last-layer-bayesian/r/last_layer_bayesian.R
```

**Refs:** Snoek, J. et al. "Scalable Bayesian optimization using deep neural networks." *ICML*, 2015; Kristiadi, A., Hein, M. & Hennig, P. "Being Bayesian, even just a bit, fixes overconfidence in ReLU networks." *ICML*, 2020; Liu, J. et al. "Simple and principled uncertainty estimation with deterministic deep learning via distance awareness (SNGP)." *NeurIPS*, 2020.

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

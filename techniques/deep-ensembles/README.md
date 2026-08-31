# Deep Ensembles (Reference Ch 29 Uncertainty Quantification)

Train **K neural networks** from different random initialisations on the
same data; average their predictive distributions. Introduced by
Lakshminarayanan, Pritzel & Blundell (2017) as a **simple** competitor to
Bayesian neural nets — often matching or beating BNN calibration in
practice.

## Decomposition

Each member outputs a Gaussian `(μ_k(x), σ_k²(x))` via the Nix-Weigend head.
The ensemble predictive distribution is a mixture of Gaussians, with

```
μ(x)     = (1/K) Σ_k μ_k(x)
Var(x)   = (1/K) Σ_k σ_k²(x)  +  (1/K) Σ_k (μ_k(x) − μ(x))²
              └── aleatoric ──┘    └────── epistemic ──────┘
```

- **Aleatoric** — data noise; irreducible; captured by each member's `σ_k²`.
- **Epistemic** — model uncertainty; shrinks with more data; captured by
  the **disagreement** across ensemble members.

## Why it works

- Different random inits find different local minima of the highly non-convex
  loss landscape → the members disagree exactly where the data are sparse.
- Cheap parallel training (each member is independent).
- No approximate-inference hyperparameters (no variational posterior, no MCMC).

## When to use

- **Any high-stakes prediction** where a calibrated uncertainty band matters
  (medical imaging, autonomous driving, molecular property prediction).
- **Detecting distribution shift / OOD** — epistemic variance spikes on
  inputs unlike the training set.
- **Model selection at deploy time** — average predictions from a small K,
  ship the ensemble.

## When NOT to use

- **Compute-bound settings** — K× the training cost and K× the inference cost.
  Options: MC-dropout, SWAG, snapshot ensembles.
- **Well-calibrated tabular ML** — often a single XGBoost + isotonic
  calibration is comparable.

## Files

- `python/deep_ensembles.py` — from-scratch K-member MLP with Gaussian-NLL
  head; explicit **aleatoric + epistemic** decomposition. Demo on noisy
  `sin(2x)` on `[-2, 2]` with test grid extended to `[-3, 3]`:
  **epistemic sd ratio (out-of-support / in-support) ≈ 3.1×**, exactly the
  intended behaviour.
- `r/deep_ensembles.R` — `reticulate` + PyTorch / TensorFlow-Probability;
  R alternatives: `torch` R port, `brulee` / `tabnet` with `workflowsets`.

## Assumptions & caveats

- **K matters** — practical papers use K = 5–10. Diminishing returns beyond
  that; 5 is the standard baseline.
- **Same architecture, same data, different inits** is the recipe. Adding
  bagging (bootstrap resampling) usually **hurts** ensemble diversity for
  neural nets.
- **Overconfident members** can still make the mixture overconfident;
  temperature-scale the ensemble output on a held-out set.
- **Aleatoric variance** requires the model to have a variance head; a
  point-prediction ensemble captures **only** epistemic.
- **Members share failure modes** on covariate shift they all trained past
  — deep ensembles are not a panacea against distribution shift.

## Related in this repo

- `mc-dropout` — Bayesian interpretation of dropout at inference time (a
  cheap ensemble).
- `bayesian-neural-network` — variational BNN reference.
- `swag`, `last-layer-bayesian` — cheaper alternatives to full ensembles.
- `epistemic-aleatoric` — the two-variance decomposition made explicit.
- `calibration-scaling`, `conformal-prediction` — post-hoc calibration on
  top of an ensemble.
- `deep-mlp-backprop`, `dropout-batchnorm` — the underlying architecture.

## Run

```
python techniques/deep-ensembles/python/deep_ensembles.py
Rscript techniques/deep-ensembles/r/deep_ensembles.R
```

**Refs:** Lakshminarayanan, B., Pritzel, A. & Blundell, C. "Simple and scalable predictive uncertainty estimation using deep ensembles." *NeurIPS*, 2017; Nix, D.A. & Weigend, A.S. "Estimating the mean and variance of the target probability distribution." *IEEE ICNN*, 1994.

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

# SWAG — SWA-Gaussian (Reference Ch 29 Uncertainty Quantification)

**Cheap Gaussian posterior over a neural net's weights**, formed by
collecting SGD iterates after warm-up and fitting a Gaussian. Maddox et
al. (2019) show it delivers deep-ensemble-quality uncertainty at a
fraction of the cost.

## Algorithm

```
1. Train the network to a low-loss basin (standard SGD).
2. Continue SGD with a constant moderately-high learning rate; every
   `stride` steps snapshot the weight vector θ_k.  Collect K snapshots.
3. Fit a Gaussian:
       μ_SWA = mean(θ_k)
       Σ_diag    = diag( mean(θ_k²) − μ_SWA² )
       Σ_lowrank = D Dᵀ / (K − 1)     with columns D_k = θ_k − μ_running(k)
       Σ         = ½ · Σ_diag + ½ · Σ_lowrank
4. At test time sample θ_s ~ N(μ_SWA, Σ) and average the predictions.
```

The **low-rank + diagonal** factorisation stores only `2K · P` numbers
(P = # weights) — no dense `P × P` covariance ever formed.

## When to use

- **Cheap alternative to a deep ensemble** — one train run + K SGD steps.
- **You already have SWA weights** (a common regulariser); SWAG is a
  one-line extension.
- **Any regression / classification** where predictive intervals help.

## When NOT to use

- **Tiny models** — the Gaussian assumption over K ~ 30 iterates is
  crude for very few parameters (the from-scratch demo here has only 5
  weights; SWAG really shines with millions).
- **Multimodal loss landscapes** — a single Gaussian collapses the
  posterior onto one basin; deep ensembles capture multiple basins.

## Files

- `python/swag.py` — from-scratch: SGD iterates on a 5-basis-function
  regressor, fit the low-rank + diagonal Gaussian, sample K posterior
  weight draws, produce a predictive band. Demo on noisy sinusoid:
  **predictive sd ratio (out-of-support / in-support) ≈ 6×**.
- `r/swag.R` — `reticulate` + PyTorch / Pyro; the R `torch` port supports
  the same iterate-collection recipe manually.

## Assumptions & caveats

- **Constant lr required in phase 2** — a decaying lr contracts the
  iterates and shrinks Σ to zero.
- **K ≥ 20** in practice; too few snapshots ⇒ noisy Σ.
- **Sampling scale** — the factor of ½ on each covariance piece is the
  paper's default; some implementations use a `scale` hyperparameter to
  temper the posterior at test time.
- **BatchNorm running stats** must be recomputed per posterior sample
  (missing this step is a common bug).
- **Correlates with the last learning rate** — larger constant lr ⇒
  wider Σ.

## Related in this repo

- `deep-ensembles`, `mc-dropout`, `bayesian-neural-network`,
  `last-layer-bayesian` — the family of approximate-posterior UQ methods.
- `sgd-momentum` — the underlying optimiser.
- `epistemic-aleatoric` — the two-variance decomposition SWAG feeds into.

## Run

```
python techniques/swag/python/swag.py
Rscript techniques/swag/r/swag.R
```

**Refs:** Maddox, W.J., Izmailov, P., Garipov, T., Vetrov, D. & Wilson, A.G. "A simple baseline for Bayesian uncertainty in deep learning." *NeurIPS*, 2019; Izmailov, P. et al. "Averaging weights leads to wider optima and better generalisation." *UAI*, 2018.

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

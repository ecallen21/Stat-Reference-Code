# Bayesian Neural Network (Reference Ch 29 Uncertainty Quantification)

Place a **prior on every weight** and infer a posterior. Full HMC is
prohibitive for modern nets, so practical BNNs use **variational
inference** (VI) or **Laplace approximations** at the last layer.

## Mean-field variational inference (Blundell 2015)

Approximate the true posterior `p(w | D)` by a factorised Gaussian
`q(w) = ∏ N(μᵢ, σᵢ²)` and minimise the ELBO:

```
L(μ, σ) = E_q[log p(D | w)] − KL(q(w) ‖ p(w))
```

The **KL to a standard-normal prior** has a closed form:

```
KL(q ‖ N(0, I)) = 0.5 Σ (σᵢ² + μᵢ² − 1 − 2 log σᵢ)
```

**Local reparameterisation** (Kingma-Salimans-Welling 2015) samples the
pre-activation `z = μ_z + σ_z · ε` directly (with `μ_z = μ_W x`, `σ_z² =
σ_W² x²`) for cheap, low-variance gradients — this is **Bayes by
Backprop**.

## Predictive distribution

```
p(y | x*, D) ≈ (1/T) Σ_t p(y | x*, w_t),    w_t ~ q(w)
```

For regression this is a mixture of Gaussians; report `mean ± sd`.

## When to use

- **Safety-critical predictions** where a calibrated posterior over
  weights matters more than the point estimate.
- **Small-to-medium networks** — VI struggles beyond a few tens of
  millions of parameters.
- **Active learning** — pick inputs that maximise BALD (Bayesian Active
  Learning by Disagreement).

## When NOT to use

- **Huge models** — deep ensembles or MC-dropout are cheaper and often
  better calibrated.
- **You need the true posterior** — mean-field VI is famously overconfident
  (underestimates posterior variance); use HMC or normalising-flow
  posteriors if you can afford it.

## Files

- `python/bayesian_neural_network.py` — from-scratch **mean-field VI** on
  a single-hidden-layer regressor with local reparameterisation and analytic
  Gaussian KL. Demo on noisy `sin(2x)`; **T = 300** MC posterior samples;
  epistemic sd ratio (out-of-support / in-support) ≈ 2.3×.
- `r/bayesian_neural_network.R` — `reticulate` + `pyro` / `TFP` /
  `bayesian-torch`; NUTS via `numpyro`.

## Assumptions & caveats

- **Mean-field factorisation** ignores posterior correlations between
  weights; posteriors are too narrow.
- **Prior choice matters** — a `N(0, 1)` prior on ReLU weights is *not*
  scale-free; Bayesian tempering (`KL·β`) is standard.
- **Warm-up needed** — start with `β ≈ 0.01` and anneal towards 1.
- **KL vs likelihood scale** — the KL should be divided by `n_train` when
  using minibatches to match the correct ELBO expectation.
- **Compare against a deep ensemble** — for a fixed compute budget the
  ensemble usually wins on calibration.

## Related in this repo

- `deep-ensembles`, `mc-dropout`, `swag`, `last-layer-bayesian` — cheaper
  uncertainty alternatives.
- `laplace-approximation` — post-hoc Laplace on the trained weights.
- `hmc-nuts`, `variational-inference` — the inference engines used to fit BNNs.
- `epistemic-aleatoric` — the two-variance decomposition.

## Run

```
python techniques/bayesian-neural-network/python/bayesian_neural_network.py
Rscript techniques/bayesian-neural-network/r/bayesian_neural_network.R
```

**Refs:** Blundell, C. et al. "Weight uncertainty in neural networks." *ICML*, 2015; Kingma, D.P., Salimans, T. & Welling, M. "Variational dropout and the local reparameterization trick." *NeurIPS*, 2015; MacKay, D. "A practical Bayesian framework for backpropagation networks." *Neural Computation*, 1992.

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

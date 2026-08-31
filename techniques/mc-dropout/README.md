# MC Dropout (Reference Ch 29 Uncertainty Quantification)

**Monte-Carlo dropout** is a cheap Bayesian approximation for neural
networks — keep dropout switched **ON at inference time** and average `T`
stochastic forward passes.

## Formula

For each test input `x`, sample `T` masks `m₁..m_T`:

```
μ(x)  = (1/T) Σ_t  f̂(x; m_t)
Var(x) = τ⁻¹  +  (1/T) Σ_t f̂(x; m_t)²  −  μ(x)²
              └─ prior aleatoric ─┘   └── epistemic (sample variance) ──┘
```

Gal & Ghahramani (2016) showed that a dropout network trained with L2
regularisation and squared-error loss is equivalent to a **variational
approximation of a deep Gaussian process** posterior, so the sample
variance across MC forward passes is an estimator of predictive
uncertainty.

## Why it works

- **No architectural change** — you already have dropout for regularisation;
  just leave it on at inference.
- **No extra parameters** — variance comes from stochastic masks.
- **T controls fidelity** — typical T = 50–200.

## When to use

- **Any dropout-trained network** where you want predictive intervals
  without retraining. Standard in medical-imaging segmentation (Kendall &
  Gal 2017).
- **Cheap baseline** to compare against deep ensembles or full BNNs.
- **Active learning** — pick the input with highest MC variance for
  labelling.

## When it's shakier

- **Concrete dropout / heteroscedastic head** — the fixed dropout rate `p`
  is a hyperparameter; concrete-dropout (Gal 2017) learns it per layer.
- **Modern architectures with BatchNorm** — combining dropout + batch norm
  is fragile; prefer just before the last dense layer.
- **Small MC samples underestimate uncertainty** — noisy for T < 20.

## Files

- `python/mc_dropout.py` — from-scratch 2-layer MLP with inverted-dropout
  masks kept alive at inference; **T = 200** predictions per test point.
  Demo on noisy `sin(2x)` train ∈ `[-2, 2]` extrapolated to `[-3, 3]`:
  **epistemic sd ratio (out-of-support / in-support) ≈ 2.3×**.
- `r/mc_dropout.R` — `reticulate` + PyTorch / TF-Probability, plus notes
  for the R `torch` port (set `mode="train"` at inference).

## Assumptions & caveats

- **Rate `p`** — too small ⇒ collapsed variance (all masks look the same);
  too large ⇒ under-fits.
- **Prior precision `τ`** — sets the aleatoric floor; often set from the
  L2 weight decay: `τ = p·l² / (2·n·λ)`.
- **Correlation across MC samples** — noisy for small `T`; batch the
  samples to amortise the fixed cost.
- **Bayesian interpretation is approximate** — MC dropout is not the true
  posterior; use with a light-touch heart.
- **Cheaper than an ensemble, but usually less well-calibrated**.

## Related in this repo

- `deep-ensembles`, `bayesian-neural-network`, `swag` — other predictive-
  uncertainty families.
- `dropout-batchnorm` — the *training-time* dropout used as a regulariser.
- `epistemic-aleatoric` — the two-variance decomposition made explicit.
- `active-learning` — a natural downstream user of MC-dropout variance.

## Run

```
python techniques/mc-dropout/python/mc_dropout.py
Rscript techniques/mc-dropout/r/mc_dropout.R
```

**Refs:** Gal, Y. & Ghahramani, Z. "Dropout as a Bayesian approximation: representing model uncertainty in deep learning." *ICML*, 2016; Kendall, A. & Gal, Y. "What uncertainties do we need in Bayesian deep learning for computer vision?" *NeurIPS*, 2017.

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

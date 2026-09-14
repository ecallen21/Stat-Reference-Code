# Gumbel-Softmax / Concrete Distribution (Reference §47.315)

Jang, Gu & Poole (2017); Maddison, Mnih & Teh (2017).
Continuous relaxation of a CATEGORICAL sample:

```
g_i ~ Gumbel(0, 1)
y_i = softmax((log α_i + g_i) / τ)
```

As τ → 0, y approaches a one-hot argmax (still differentiable
via the relaxation). Enables end-to-end training through
discrete stochastic layers (categorical VAEs, routing
networks, mixture-of-experts gates).

## Files

- `python/gumbel_softmax_relaxation.py` — Gumbel sampling +
  softmax. Categorical probs [0.1, 0.3, 0.6], 20 000 samples.
  As τ decreases 5 → 1 → 0.5 → 0.1, avg top-value rises from
  ~0.42 to ~0.94, and the empirical categorical stabilises
  at the true probs. Straight-through variant illustrated.
- `r/gumbel_softmax_relaxation.R` — `torch` (R), reticulate
  + `torch` (R);
  `torch.nn.functional.gumbel_softmax`, TFP
  `RelaxedOneHotCategorical`, Pyro / NumPyro, from-scratch
  (Python).

## When to use

- **Categorical latent variables** in VAEs / RL.
- **Differentiable routing** — Mixture of Experts,
  hard-attention.
- **Structured prediction** — differentiable subset / graph
  sampling.

## When NOT to use

- **Very small logit magnitudes** — Gumbel noise dominates,
  everything looks uniform.
- **When only argmax is used** — straight-through is
  simpler than a proper Gumbel softmax.
- **Numerically unstable regimes** — very small τ → gradient
  norms diverge.

## Assumptions & caveats

- **τ annealing** — start large (τ = 1-5), anneal toward τ
  ≈ 0.1 during training.
- **Straight-through variant** — forward hard argmax,
  backward soft (biased but low variance).
- **Score-function estimator (REINFORCE)** — the
  unbiased-but-high-variance alternative.
- **Compare against REBAR / RELAX** — modern low-variance
  gradient estimators.

## Related in this repo

- `straight-through-estimator` — companion discrete-gradient
  trick.
- `variational-autoencoder`, `beta-vae-disentangle`,
  `iwae-importance-weighted` — VAE families that use it.
- `mixture-of-experts` — routing that benefits from Gumbel
  softmax.
- `reinforcement-learning-basics`, `ppo-clipped` —
  score-function alternatives.

## Run

```
python techniques/gumbel-softmax-relaxation/python/gumbel_softmax_relaxation.py
Rscript techniques/gumbel-softmax-relaxation/r/gumbel_softmax_relaxation.R
```

**Refs:** Jang, E., Gu, S. and Poole, B. "Categorical reparameterization with Gumbel-Softmax." In *ICLR*, 2017; Maddison, C.J., Mnih, A. and Teh, Y.W. "The Concrete distribution: A continuous relaxation of discrete random variables." In *ICLR*, 2017.

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

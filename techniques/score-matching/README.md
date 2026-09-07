# Score Matching (Reference §46.17)

Hyvärinen (2005). Estimates the **score** of a model density
`ψ_model(x) = ∇_x log p(x; θ)` by minimising

    J(θ) = ½ · E[ ‖ψ_model(x) − ψ_data(x)‖² ]

which via integration by parts (Hyvärinen's identity) becomes

    J(θ) = E[ ½ · ‖ψ(x)‖² + Σⱼ ∂ψⱼ/∂xⱼ ]

so the **normalising constant `log Z(θ)` drops out**. Fit θ by
minimising the empirical `J` on samples from `p_data`.

## Related

- **Sliced score matching** — projections for high-dim.
- **Denoising score matching** — Song & Ermon 2019; backbone of
  diffusion models.
- **Ratio matching** — Gutmann-Hyvärinen NCE cousin.

## Files

- `python/score_matching.py` — Hyvärinen's identity applied to a
  1-D Gaussian from scratch. Demo (n=1000, N(2.0, 1.5)): score
  matching recovers (μ̂, σ̂) = (1.928, 1.465), matching MLE
  (1.928, 1.466) as expected for a Gaussian family.
- `r/score_matching.R` — no dedicated CRAN implementation; describes
  Hyvärinen's identity and points to deep-net variants (torch,
  score-sde).

## When to use

- **Unnormalised densities / energy-based models** — Z intractable.
- **Continuous data on ℝⁿ** — score matching's identity requires
  differentiable log-density and vanishing boundary terms.
- **Deep score networks** — sliced / denoising SM trains
  score networks for diffusion generative models.

## When NOT to use

- **Discrete data** — the identity breaks; use ratio matching or
  NCE.
- **Densities with bounded support and non-vanishing tails** —
  boundary terms fail to cancel; use sm4mb-style extensions.
- **You need the normalising constant** — SM does not estimate `Z`;
  use bridge sampling or MLE.

## Assumptions & caveats

- **Differentiable log-density** with vanishing tails.
- **Enough samples** — SM has larger variance than MLE for models
  where MLE is tractable (Vincent 2011); use only when MLE isn't.
- **Automatic differentiation** — deep-net SM needs `d ψ / d x`;
  use torch / jax.
- **Bias-variance of estimators** — denoising SM introduces noise σ
  that must be scheduled (e.g. NCSN, EDM).

## Related in this repo

- `energy-based-models`, `diffusion-model`, `variational-autoencoder`,
  `normalizing-flows` — generative-model neighbours.
- `kl-divergence`, `f-divergences` — objective-function cousins.
- `bridge-sampling-evidence` — complement for estimating `Z`.

## Run

```
python techniques/score-matching/python/score_matching.py
Rscript techniques/score-matching/r/score_matching.R
```

**Refs:** Hyvärinen, A. "Estimation of non-normalized statistical models by score matching." *JMLR*, 6: 695-709, 2005; Song, Y. & Ermon, S. "Generative modeling by estimating gradients of the data distribution." *NeurIPS*, 2019.

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

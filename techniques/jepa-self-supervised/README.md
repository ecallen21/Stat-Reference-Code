# JEPA -- Joint Embedding Predictive Architecture (Reference §47.23)

LeCun (2022); Assran et al. (I-JEPA, 2023). Self-supervised learning
that predicts **latent representations** of a target region from a
context region — no pixel-level reconstruction (as in autoencoders /
MAE) and no generative denoising (as in diffusion).

## Setup

- `x` : full input (image / audio / video segment).
- `x_ctx, x_tgt` : disjoint context / target views (crops / masks).
- `f_θ, f_ζ` : student and (EMA) target encoders.
- `g_φ` : latent-space predictor.

Loss (I-JEPA):

    L = ‖ g_φ(f_θ(x_ctx))  −  f_ζ(x_tgt) ‖₂²

Anti-collapse comes from:

- **EMA target encoder** (BYOL-style momentum update).
- **Predictor bottleneck.**
- **Multi-target, multi-context masking.**
- **VICReg-style variance / covariance regularisation** (V-JEPA).

## Files

- `python/jepa_self_supervised.py` — toy 1-D JEPA from scratch:
  linear encoders, linear predictor, EMA target update. Demo: loss
  drops from 1.79 → 0.47 → 0.25 across 400 epochs. Prints
  eigenvalue spread of context / target reps; a healthy JEPA keeps
  `min_eig > 0` (no total collapse).
- `r/jepa_self_supervised.R` — no R; describes Meta's I-JEPA and
  V-JEPA repos and adjacent SSL methods (VICReg, DINOv2).

## When to use

- **Large unlabelled corpora** — images, video, audio, code — where
  pixel-level reconstruction spends capacity on unimportant detail.
- **Downstream transfer** — JEPA reps transfer well to
  classification, detection, segmentation.
- **Video / temporal SSL** — V-JEPA predicts future / masked frames'
  representations without generation.

## When NOT to use

- **Generation is the goal** — use MAE / diffusion / autoregressive
  models; JEPA gives features, not samples.
- **Small labelled datasets are enough** — supervised training beats
  SSL when you have plenty of labels.
- **Extremely small models** — the EMA teacher + predictor pair
  overhead can outweigh the SSL gain.

## Assumptions & caveats

- **Representation collapse** — the biggest risk; use EMA + variance
  regularisation + big predictor.
- **Masking strategy** — non-trivial: I-JEPA uses multi-block
  masking with predictor prompts encoding target positions.
- **Compute budget** — SSL needs many epochs; JEPA is usually
  cheaper than pixel-reconstruction but still large.
- **Evaluation** — measure by linear-probe or fine-tune on
  downstream tasks; the SSL loss alone is not comparable across
  runs.

## Related in this repo

- `autoencoder`, `variational-autoencoder`, `diffusion-model` — the
  reconstructive / generative SSL alternatives.
- `contrastive-learning`, `masked-language-modeling` — sibling SSL
  paradigms.
- `transformer-encoder`, `vision-transformer` — typical backbones
  for JEPA.
- `knowledge-distillation` — the EMA-teacher pattern reappears here.

## Run

```
python techniques/jepa-self-supervised/python/jepa_self_supervised.py
Rscript techniques/jepa-self-supervised/r/jepa_self_supervised.R
```

**Refs:** LeCun, Y. "A path towards autonomous machine intelligence." *Meta AI open review*, 2022; Assran, M. et al. "Self-supervised learning from images with a joint-embedding predictive architecture" (I-JEPA). *CVPR*, 2023; Bardes, A. et al. "Revisiting feature prediction for learning visual representations from video" (V-JEPA), 2024.

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

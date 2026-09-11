# MAE — Masked Autoencoders (Reference §47.228)

He, Chen, Xie, Li, Dollár & Girshick (2022, CVPR). Asymmetric
encoder-decoder ViT:

1. Divide image into patches; randomly MASK 75 % of patches.
2. **Encoder** processes only VISIBLE 25 % of patches.
3. **Decoder** (shallow) reconstructs pixel values of masked
   patches from encoder outputs + [MASK] positional embeddings.
4. Loss = MSE on MASKED patches only.

75 % mask ratio is high vs BERT's 15 % because images are more
locally-redundant. Enables scaling to ViT-Huge on ImageNet with
strong linear-probe / fine-tune performance.

## Files

- `python/mae_masked_autoencoders.py` — toy MAE with linear
  encoder / decoder on a 32×32 checkerboard-plus-gradient image
  (16 patches):
  - 15 % mask → recon MSE 0.97.
  - 50 % mask → 0.96.
  - **75 % mask (paper default)** → 0.81 (best of tested).
  - 90 % mask → 0.81.
- `r/mae_masked_autoencoders.R` — recommends
  `facebookresearch/mae`, `timm`.

## When to use

- **Vision transformer pretraining** at scale — encoder-only
  reduces compute vs contrastive.
- **When strong linear probes** are needed post-pretraining.
- **Combining with CLIP-style text supervision** (paper's follow-
  ups).

## When NOT to use

- **Very small datasets** — MAE benefits from millions of images.
- **When lightweight (CNN) models suffice** for the downstream
  task.
- **Fine-grained pixel-perfect reconstruction** — decoder is
  intentionally weak.

## Assumptions & caveats

- **Mask ratio 75 %** essential — lower ratios don't force
  content-level reasoning.
- **Shallow decoder** — 4-8 blocks; much smaller than encoder.
- **Loss only on masked patches** — visible-patch reconstruction
  is trivial.
- **Position embeddings** for both visible and mask tokens.

## Related in this repo

- `vision-transformer-vit` — the encoder backbone.
- `dino-self-supervised-vision`, `simclr-contrastive`,
  `barlow-twins` — SSL vision cousins.
- `autoencoder`, `variational-autoencoder` — encoder-decoder
  ancestors.

## Run

```
python techniques/mae-masked-autoencoders/python/mae_masked_autoencoders.py
Rscript techniques/mae-masked-autoencoders/r/mae_masked_autoencoders.R
```

**Refs:** He, K. et al. "Masked autoencoders are scalable vision learners." *CVPR*, 2022.

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

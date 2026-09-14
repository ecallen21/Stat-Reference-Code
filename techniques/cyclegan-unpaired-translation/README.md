# CycleGAN — Unpaired Image Translation (Reference §47.287)

Zhu, Park, Isola & Efros (2017). Learn a mapping `G: X → Y`
between two domains WITHOUT paired examples using a
CYCLE CONSISTENCY loss:

```
L_GAN(G, D_Y) + L_GAN(F, D_X)                        (adversarial)
  + λ · (‖F(G(x)) − x‖₁ + ‖G(F(y)) − y‖₁)             (cycle)
```

Two generators (G, F) + two discriminators (D_X, D_Y) trained
jointly. Enables horse↔zebra, photo↔painting, summer↔winter
style transfer without paired data.

## Files

- `python/cyclegan_unpaired_translation.py` — 1-D toy: shift
  X ~ N(0, 1) to Y ~ N(2, 1). Cycle-consistency loss is
  ZERO whenever θ_F = −θ_G — the shift is unidentified by
  cycle alone. Adversarial distribution-matching loss is
  minimised at θ = 2. Illustrates why both losses are needed.
- `r/cyclegan_unpaired_translation.R` — reticulate + PyTorch
  CycleGAN (R); junyanz/pytorch-CycleGAN-and-pix2pix, tf-gan,
  from-scratch (Python).

## When to use

- **Unpaired image translation** — horse↔zebra style,
  photo↔painting, medical modality transfer.
- **Domain adaptation** — pseudo-labels via cycle
  consistency.
- **Data augmentation** across domains.

## When NOT to use

- **Paired examples available** — pix2pix / U-Net supervised
  training is stronger.
- **Content structure not preserved by cycle** — CycleGAN
  can hallucinate content (Chu et al 2017 "steganography of
  CycleGAN").
- **Semantic content changes** — cycle assumes preservable
  structure; drastic content flips (add/remove objects)
  need conditional models.

## Assumptions & caveats

- **λ (cycle weight)** — 10 is the paper default; larger
  = more content preservation, less style change.
- **Identity loss** — `‖G(y) − y‖₁` added in the paper for
  colour preservation.
- **Buffered discriminator memory** — 50-image replay
  buffer stabilises D training.
- **Mode collapse** — same as any GAN; check output diversity.

## Related in this repo

- `wgan-gp-gradient-penalty`, `gan-training`,
  `conditional-gan-cgan` — related GAN variants.
- `stylegan` (via cleaner variant), `diffusion-model` —
  alternative generative approaches.
- `beta-vae-disentangle` — non-adversarial cousin.
- `contrastive-learning`, `siamese-networks` — different
  unpaired-learning setups.

## Run

```
python techniques/cyclegan-unpaired-translation/python/cyclegan_unpaired_translation.py
Rscript techniques/cyclegan-unpaired-translation/r/cyclegan_unpaired_translation.R
```

**Refs:** Zhu, J.-Y., Park, T., Isola, P. and Efros, A.A. "Unpaired image-to-image translation using cycle-consistent adversarial networks." In *ICCV*, pp. 2242-2251, 2017.

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

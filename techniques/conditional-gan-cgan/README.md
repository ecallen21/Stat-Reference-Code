# Conditional GAN (cGAN) (Reference §47.108)

Mirza & Osindero (2014). Extends the vanilla GAN by conditioning
both G and D on a label `y`:

    min_G max_D  E[log D(x, y)] + E[log(1 − D(G(z, y), y))].

Foundation of class-conditional generation, Pix2Pix, and
conditional Stable Diffusion.

## Files

- `python/conditional_gan_cgan.py` — from-scratch **equilibrium
  demo** (full adversarial training needs a DL framework and is
  fiddly; the equilibrium is what matters analytically). Demo
  (n=4000, 1-D two-mode target, y∈{0, 1}):
  - Untrained G → D distinguishes real vs fake at **acc 0.51**.
  - Optimum G (MLE per class) → D at **acc 0.500** (equilibrium).
  - Generated samples match target N(±2, 0.3²) per class.
- `r/conditional_gan_cgan.R` — no first-class R port;
  `torchvision`, `torchGAN`, `pytorch-lightning` (Python).

## When to use

- **Class-conditional image / text / audio generation**.
- **Image-to-image translation** — Pix2Pix conditions on a source
  image.
- **Data augmentation** — synthesise rare-class examples.
- **Fairness / balancing** — generate under-represented subgroup
  data.

## When NOT to use

- **When density estimates** rather than samples are needed —
  normalising flows.
- **Small datasets** — GAN training unstable; VAE or flow more
  data-efficient.
- **Verifiable / calibrated probabilities** required — GANs give
  samples, not probabilities.
- **Regulated privacy contexts** — GAN samples can memorise
  training data.

## Assumptions & caveats

- **Adversarial training instability** — mode collapse, oscillation.
  Solutions: WGAN, spectral norm, TTUR, feature matching.
- **Class conditioning** requires informative y that G can use.
- **Evaluation** hard — no clean likelihood; use FID, IS, precision-
  recall, or downstream classifier accuracy.
- **Distribution shift** — G may exploit weak D and produce
  unrealistic samples that fool D but not humans.

## Related in this repo

- `gan-training`, `wgan-wasserstein-gan`, `diffusion-model`,
  `variational-autoencoder`, `normalizing-flows`,
  `energy-based-models` — generative model family.
- `contrastive-learning`, `byol-simsiam`,
  `contrastive-predictive-coding`, `jepa-self-supervised` — SSL
  neighbours.
- `fair-representations-lfr`, `adversarial-debiasing` — adversarial
  training analogues for fairness.
- `mmd-two-sample-test` — an alternative distribution-matching
  loss.

## Run

```
python techniques/conditional-gan-cgan/python/conditional_gan_cgan.py
Rscript techniques/conditional-gan-cgan/r/conditional_gan_cgan.R
```

**Refs:** Mirza, M. & Osindero, S. "Conditional Generative Adversarial Nets." *arXiv:1411.1784*, 2014; Goodfellow, I.J. et al. "Generative Adversarial Nets." *NeurIPS*, 2014.

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

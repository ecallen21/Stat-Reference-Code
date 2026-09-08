# BYOL / SimSiam (Reference §47.50)

Grill et al 2020 'Bootstrap Your Own Latent'; Chen & He 2021
'Exploring Simple Siamese Representation Learning'. Contrastive-
free self-supervised learning — learn image representations
WITHOUT explicit negative examples:

    z_1 = f_θ(view_1)                (online encoder)
    z_2 = f_ξ(view_2)                (target encoder, EMA of online)
    p_1 = q(z_1)                     (predictor MLP on online branch)
    L    = 2 − 2 · cos(p_1, stop_grad(z_2))

SimSiam ablates the EMA target (uses hard `stop_grad` only) and
still avoids collapse, proving that the stop-gradient — not the
EMA — is the key regulariser.

## Files

- `python/byol_simsiam.py` — toy 1-layer online + EMA target +
  linear predictor, numerical grad on the online weights. Demo
  (n=64, d_in=4, d_out=8, 40 steps):
  loss drops 1.26 → 0.25; feature std ≈ 0.25 (no collapse).
- `r/byol_simsiam.R` — no R equivalent; `lightly`, `solo-learn`,
  or `torchvision` in Python.

## When to use

- **Label-scarce vision domains** — medical imaging, satellite,
  microscopy — pre-train on unlabelled corpora.
- **Domain adaptation** — self-supervised features often transfer
  better than supervised.
- **When negatives are hard to define** — segmentation, dense
  tasks.

## When NOT to use

- **Small datasets** — needs many augmentations per epoch to work.
- **Text / discrete data** — MLM / contrastive objectives (SimCSE,
  BERT) fit better.
- **Downstream needs fine control** — linear probes on BYOL
  features may plateau below supervised.

## Assumptions & caveats

- **Augmentations** are the training signal — poor augmentations
  → collapsed or unhelpful features.
- **Stop-gradient** is essential; removing it → constant
  representation.
- **EMA momentum** typically 0.996–0.999 for BYOL; SimSiam uses
  none.
- **Batch normalisation** subtleties debated — DeCLIP and others
  reproduce BYOL without BN by explicit whitening.

## Related in this repo

- `contrastive-learning`, `jepa-self-supervised` — SSL cousins.
- `autoencoder`, `variational-autoencoder`, `masked-language-
  modeling` — self-supervision alternatives.
- `knowledge-distillation`, `mean-teacher` (see `semi-supervised-
  pseudo-labeling`) — teacher-student ideas.
- `transfer-learning`, `lora-peft` — downstream adaptation of
  pre-trained features.

## Run

```
python techniques/byol-simsiam/python/byol_simsiam.py
Rscript techniques/byol-simsiam/r/byol_simsiam.R
```

**Refs:** Grill, J.-B. et al. "Bootstrap your own latent: A new approach to self-supervised learning." *NeurIPS*, 2020; Chen, X. & He, K. "Exploring simple siamese representation learning." *CVPR*, 2021.

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

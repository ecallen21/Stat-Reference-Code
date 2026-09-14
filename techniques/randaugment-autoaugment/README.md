# RandAugment / AutoAugment (Reference §47.304)

Cubuk et al (2019, AutoAugment); Cubuk et al (2020,
RandAugment). Data augmentation policies for image
classification:

- **AutoAugment**: RL-searched policy of (op, prob, magnitude)
  triples optimised on a proxy task.
- **RandAugment**: `N` transforms drawn uniformly from a fixed
  pool, all applied with a single global magnitude `M`.

RandAugment removes AutoAugment's expensive search yet matches
downstream accuracy. Standard for CIFAR / ImageNet training.

## Files

- `python/randaugment_autoaugment.py` — 5-op RandAugment
  simulator (rotate, shear-x, contrast, brightness, solarize).
  Demo across N ∈ {1, 2, 3} and M ∈ {2, 5, 9} shows how
  cumulative applied-op sequences and pixel statistics shift.
- `r/randaugment_autoaugment.R` — reticulate +
  `torchvision.transforms.v2` (R); `torchvision.RandAugment`,
  `timm.data.auto_augment`, `albumentations`, from-scratch
  (Python).

## When to use

- **Image classification** training — RandAugment is a
  default choice.
- **Detection / segmentation** — combine with mosaic /
  MixUp / CutMix.
- **Fine-tuning small datasets** — augmentation boosts
  generalisation.

## When NOT to use

- **When augmentation destroys label semantics** (e.g. OCR
  where rotation flips 6↔9).
- **For pretraining self-supervised models** where augmentation
  choice is already carefully tuned (SimCLR / BYOL / DINO
  have their own recipes).
- **Very small M (≈ 1)** — barely different from no
  augmentation.

## Assumptions & caveats

- **N, M hyperparameters** — CIFAR N=2, M=14 is standard;
  ImageNet N=2, M=9.
- **Operation pool** — 14 canonical ops in the original
  paper; libraries add / remove some.
- **TrivialAugment** (Müller & Hutter 2021) uses ONE op with
  a random magnitude per image — even simpler and often as
  good.
- **AugMix** blends multiple augmentation chains for
  robustness to distribution shift.

## Related in this repo

- `mixup`, `cutmix` — complementary augmentation strategies.
- `label-smoothing` — regulariser typically paired with heavy
  augmentation.
- `feature-squeezing`, `randomized-smoothing` — related
  adversarial-robustness augmentation ideas.
- `contrastive-learning` — SSL augmentation is central.

## Run

```
python techniques/randaugment-autoaugment/python/randaugment_autoaugment.py
Rscript techniques/randaugment-autoaugment/r/randaugment_autoaugment.R
```

**Refs:** Cubuk, E.D., Zoph, B., Mane, D., Vasudevan, V. and Le, Q.V. "AutoAugment: Learning augmentation strategies from data." In *CVPR*, 2019; Cubuk, E.D., Zoph, B., Shlens, J. and Le, Q.V. "RandAugment: Practical automated data augmentation with a reduced search space." In *NeurIPS Workshop / CVPR-W*, 2020.

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

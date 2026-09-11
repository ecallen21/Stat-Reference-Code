# DINO — Self-Distillation with No Labels (Reference §47.227)

Caron, Touvron, Misra, Jégou, Mairal, Bojanowski & Joulin (2021,
ICCV). Two networks with identical architecture (STUDENT +
TEACHER). Student sees a global crop; teacher sees ANOTHER global
crop and multiple local crops. Loss: cross-entropy on
softmax(pred / temp):

    L = −Σ_i Σ_j q_teacher_i · log p_student_j   (i ≠ j)
    teacher = EMA(student)

**Centering + sharpening** prevent collapse. DINO ViT features are
remarkable: attention heads segment objects without labels.

## Files

- `python/dino_self_supervised_vision.py` — toy student/teacher
  linear model with EMA update + centering:
  - Simulates 200 iterations of multi-crop self-distillation.
  - Student-teacher weight distance ~0.02 (tight EMA).
  - Illustrates the mechanism (real DINO uses ViT-S/B + full
    backprop).
- `r/dino_self_supervised_vision.R` — recommends
  `facebookresearch/dino`, DINOv2, `solo-learn`.

## When to use

- **Self-supervised vision pretraining** with strong linear-probe
  performance.
- **Zero-shot segmentation** — DINO attention naturally segments.
- **When labels are scarce** or expensive.

## When NOT to use

- **Small dataset** — DINO needs 100k+ images to shine.
- **When supervised is fine** — DINO adds pretraining overhead.
- **Non-image data** — DINO is specifically designed for images.

## Assumptions & caveats

- **EMA momentum τ = 0.996** typical; too small = teacher-student
  collapse.
- **Multi-crop** (2 global + 6 local) essential for local-global
  consistency.
- **Centering + sharpening** — temperature-based; without them
  the teacher's output collapses to one class.
- **DINOv2** (2023) adds iBOT-style masked-image modelling and
  scales to billions of images.

## Related in this repo

- `simclr-contrastive`, `barlow-twins`,
  `contrastive-learning`, `mae-masked-autoencoders` — SSL vision
  cousins.
- `swag`, `stochastic-weight-averaging-swa` — EMA-based training
  cousins.
- `clip-vision-language`, `sam-segment-anything` — vision
  foundation-model neighbours.

## Run

```
python techniques/dino-self-supervised-vision/python/dino_self_supervised_vision.py
Rscript techniques/dino-self-supervised-vision/r/dino_self_supervised_vision.R
```

**Refs:** Caron, M. et al. "Emerging properties in self-supervised vision transformers (DINO)." *ICCV*, 2021; Oquab, M. et al. "DINOv2: Learning robust visual features without supervision." *TMLR*, 2024.

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

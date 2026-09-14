# Dice Loss (Reference §47.298)

Milletari, Navab & Ahmadi (2016, V-Net); Sudre et al (2017).
Loss for dense prediction that directly optimises the Dice /
F1 overlap:

```
Dice = 2 · |A ∩ B| / (|A| + |B|)
Soft: (2 Σ p_i y_i + ε) / (Σ p_i + Σ y_i + ε)
L_Dice = 1 − Dice
```

Combined with BCE / cross-entropy (Dice + BCE) it mitigates
class imbalance in biomedical / satellite segmentation.

## Files

- `python/dice_loss_segmentation.py` — Dice, BCE, and
  combined losses. Demo: 100×100 mask, 1% foreground. Trivial
  all-zero predictor: BCE 0.21 (misleadingly small), Dice
  1.00 (correctly rejects). Off-by-3 box: Dice 0.32; perfect:
  Dice 0.03. Combined loss inherits imbalance-robustness
  from Dice and smooth gradients from BCE.
- `r/dice_loss_segmentation.R` — `torch` (R), reticulate +
  `smp.losses` (R); `monai.losses.DiceLoss`,
  `segmentation-models-pytorch.losses.DiceLoss`, from-scratch
  (Python).

## When to use

- **Heavy class imbalance** — biomedical lesions,
  satellite roads, defect detection.
- **Direct overlap optimisation** — instead of proxy pixel
  BCE.
- **Combined with BCE** for gradient stability.

## When NOT to use

- **Balanced classes** — plain cross-entropy is fine.
- **Very small annotation** — Dice can be unstable when
  target is tiny; use focal Tversky.
- **Sparse point predictions** — Dice fails silently when
  both masks are empty.

## Assumptions & caveats

- **Batch reduction** — mean over samples then classes, not
  the other way around; sample-Dice is more robust.
- **Squared vs linear denominator** — V-Net paper squares;
  most modern implementations do not.
- **Multiclass Dice** — mean over classes (macro Dice) is
  standard.
- **Combine, don't replace** — Dice + BCE / CE typically
  best.

## Related in this repo

- `u-net-segmentation`, `semantic-segmentation-fcn` — main
  users.
- `focal-tversky-loss` — asymmetric FP/FN alternative.
- `cross-entropy-log-loss` — the BCE baseline.
- `iou-generalized-iou` — related overlap metric.

## Run

```
python techniques/dice-loss-segmentation/python/dice_loss_segmentation.py
Rscript techniques/dice-loss-segmentation/r/dice_loss_segmentation.R
```

**Refs:** Milletari, F., Navab, N. and Ahmadi, S.-A. "V-Net: Fully convolutional neural networks for volumetric medical image segmentation." In *3DV*, 2016; Sudre, C.H. et al. "Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations." In *DLMIA*, 2017.

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

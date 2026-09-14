# Focal Tversky Loss (Reference §47.299)

Salehi et al (2017, Tversky); Abraham & Khan (2019, Focal
Tversky). Tversky generalises Dice with asymmetric FP/FN
weighting:

```
Tversky = TP / (TP + α · FN + β · FP)
Dice = special case α = β = 0.5
Focal Tversky: L = (1 − Tversky)^γ,  γ ∈ [1, 3]
```

Emphasises loss on hard examples (small structures) — key for
tiny-lesion segmentation.

## Files

- `python/focal_tversky_loss.py` — Tversky and focal Tversky
  on 3 scenarios: high FN (misses the lesion), high FP
  (over-predicts), balanced off-by-3. With α=0.7 (FN heavier),
  high FN gets Tversky 0.12 (worse than Dice 0.16) — the
  penalty steers training toward recovering missed lesions.
- `r/focal_tversky_loss.R` — `torch` (R) + custom, reticulate
  + `smp.losses` (R); `monai.losses.TverskyLoss`,
  `smp.losses.TverskyLoss`, from-scratch (Python).

## When to use

- **Small-lesion detection** where recall matters much more
  than precision.
- **Class-imbalanced segmentation** — asymmetric α, β lets
  you tune the trade-off.
- **Hard-example emphasis** via focal γ > 1.

## When NOT to use

- **Balanced classes** — plain Dice / BCE.
- **When precision and recall matter equally** — α = β = 0.5
  (Dice).
- **Very high γ** — loss saturates on easy examples but can
  destabilise training on some batches.

## Assumptions & caveats

- **α + β = 1** convention makes Tversky comparable to Dice.
- **γ in [1, 3]** — practical range; γ > 3 rarely helps.
- **Combine with BCE / CE** for gradient stability early in
  training.
- **Report all three** (Dice, Tversky, Focal Tversky) during
  ablation — no free lunch.

## Related in this repo

- `dice-loss-segmentation` — the α = β = 0.5 baseline.
- `focal-loss-imbalance` — classification analogue.
- `u-net-segmentation`, `semantic-segmentation-fcn` — main
  users.
- `class-imbalance` — overall framing.

## Run

```
python techniques/focal-tversky-loss/python/focal_tversky_loss.py
Rscript techniques/focal-tversky-loss/r/focal_tversky_loss.R
```

**Refs:** Salehi, S.S.M., Erdogmus, D. and Gholipour, A. "Tversky loss function for image segmentation using 3D fully convolutional deep networks." In *MLMI (MICCAI Workshop)*, 2017; Abraham, N. and Khan, N.M. "A novel focal Tversky loss function with improved attention U-Net for lesion segmentation." In *ISBI*, 2019.

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

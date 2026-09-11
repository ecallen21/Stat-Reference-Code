# Class-Balanced Loss (Reference §47.252)

Cui, Jia, Lin, Song & Belongie (2019). Reweight loss per class
by the EFFECTIVE NUMBER of samples:

```
E_c = (1 - beta^n_c) / (1 - beta)
w_c = 1 / E_c   (then normalise)
```

where `beta ∈ [0, 1)` controls the discount for adding one more
sample. As `beta → 1`, `w_c → 1 / n_c` (inverse-frequency); as
`beta → 0`, `w_c → 1` (uniform). Recommended `beta = (N-1)/N`.

## Files

- `python/class_balanced_loss.py` — Effective-number weight
  formula and sklearn LR demo. Demo: n=3000, 15 features,
  95/5 imbalance. `beta = 0.9999` gives min/maj weight ratio
  ≈ 15 (near inverse-freq 19); `beta = 0.9` gives ratio 1.1
  (near uniform). Plain LR F1 = 0.63; inverse-freq
  over-corrects to F1 = 0.36; class-balanced (beta=0.999)
  lands between at F1 = 0.51.
- `r/class_balanced_loss.R` — `caret(weights=...)`,
  `mlr3learners(class.weights)`, `xgboost(scale_pos_weight)`
  (R); sklearn `class_weight='balanced'`, custom PyTorch /
  TF, from-scratch (Python).

## When to use

- **Long-tailed classification** with many classes (image
  recognition, retail catalogue) where inverse-frequency
  over-weights rare classes.
- **Deep-learning training loops** — apply as per-sample or
  per-batch weight.
- **Combined with re-sampling** — CB-loss on a mildly
  oversampled dataset.

## When NOT to use

- **Nearly balanced classes** — weights collapse to ≈ 1;
  wasted machinery.
- **Very few samples per class** — `E_c` estimate unstable;
  fall back on inverse-freq or macro-F1 optimisation.
- **Where duplicate samples ARE informative** (e.g. weighted
  survey data) — effective-number assumption breaks.

## Assumptions & caveats

- **Effective-number derivation** treats each sample's
  contribution as diminishing geometrically; suits image
  features but is heuristic elsewhere.
- **Beta choice** matters — sweep on a validation split.
- **Combine with data augmentation** with caution:
  augmentations affect `n_c` in an implicit, non-multiplicative
  way.
- **Normalise** weights so mean = 1 to keep learning rate
  interpretable.

## Related in this repo

- `focal-loss-imbalance` — the sample-level companion.
- `smote-oversampling`, `adasyn-oversampling` — resampling
  routes.
- `demographic-parity`, `equal-opportunity` — fairness-aware
  cousins (reweighting for group parity).
- `class-imbalance` — the overall framing.

## Run

```
python techniques/class-balanced-loss/python/class_balanced_loss.py
Rscript techniques/class-balanced-loss/r/class_balanced_loss.R
```

**Refs:** Cui, Y., Jia, M., Lin, T.-Y., Song, Y. and Belongie, S. "Class-Balanced Loss Based on Effective Number of Samples." In *IEEE CVPR*, pp. 9268-9277, 2019.

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

# Focal Loss for Imbalance (Reference §47.251)

Lin, Goyal, Girshick, He & Dollár (2017, RetinaNet). Down-weight
easy examples so training concentrates on the hard,
minority-class ones:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

where `p_t` is the predicted probability of the TRUE class,
`gamma > 0` shrinks loss on well-classified examples, and
`alpha` rebalances classes. Standard for dense object detection
and any highly-imbalanced classification.

## Files

- `python/focal_loss_imbalance.py` — Numpy implementation with
  focal gradient for training. Demo: n=3000, 15 features, 95/5
  imbalance. Down-weighting demo: at gamma=2, `(1-p_t)^gamma`
  = 0.25 (uncertain), 0.01 (p_t=0.9), 0.0001 (p_t=0.99) —
  well-classified examples contribute <1% of gradient.
- `r/focal_loss_imbalance.R` — `torch` + custom weight,
  `keras3::loss_binary_focal_crossentropy`,
  `lightgbm(is_unbalance)` (R);
  `torchvision.ops.sigmoid_focal_loss`,
  `tf.keras.losses.BinaryFocalCrossentropy`, from-scratch
  (Python).

## When to use

- **Extreme imbalance** (< 5% positives) in deep-learning /
  gradient-boosted models where BCE saturates on easy negatives.
- **Object detection** — RetinaNet's motivating case.
- **Foreground/background segmentation** with tiny objects.
- **Fine-grained classification** where a few hard examples
  dominate.

## When NOT to use

- **Mild imbalance** — plain BCE with `pos_weight` is simpler
  and often equivalent.
- **Very noisy labels** — focal amplifies attention on
  "hard" points which may just be mislabelled.
- **Small `n`** — reweighting shrinks the effective sample.

## Assumptions & caveats

- **Choice of `gamma`** — paper recommends gamma=2, alpha=0.25;
  tune per task.
- **Calibration drift** — focal-loss outputs are miscalibrated;
  apply Platt / isotonic recalibration on a held-out set for
  probabilistic use.
- **Gradient behaviour** — easy examples still contribute
  gradient signal ∝ (1-p_t)^gamma; not zero.
- **Alpha ↔ class weight** — combining focal with class weights
  can double-count; pick one or the other.

## Related in this repo

- `class-balanced-loss` — the effective-number rebalancer.
- `smote-oversampling`, `adasyn-oversampling` — resampling
  routes to the same goal.
- `label-smoothing` — a different regularisation of
  cross-entropy.
- `cross-entropy-log-loss` — the base loss focal modifies.

## Run

```
python techniques/focal-loss-imbalance/python/focal_loss_imbalance.py
Rscript techniques/focal-loss-imbalance/r/focal_loss_imbalance.R
```

**Refs:** Lin, T.-Y., Goyal, P., Girshick, R., He, K. and Dollár, P. "Focal Loss for Dense Object Detection." In *IEEE ICCV*, pp. 2999-3007, 2017.

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

# Label Smoothing (Reference Ch 30 Robustness)

Replace one-hot targets with a small mixture of one-hot + uniform.
Introduced as a regulariser in the Inception-v3 paper (Szegedy et al.
2016) and now a default in most modern classification recipes.

## Formula

```
ỹ  =  (1 − ε) · one_hot(y)  +  ε / K
```

Cross-entropy against `ỹ` decomposes as

```
CE(ỹ, p̂)  =  (1 − ε) · CE(y, p̂)  +  ε · CE(Uniform, p̂)
```

The extra `ε · CE(Uniform, p̂)` term penalises overconfidence — the
model is pulled away from predicting `p̂_y = 1` exactly.

## What it changes

- **Lower mean confidence** on max-softmax.
- **Smaller weight norms** in the final layer.
- **Better calibration** when the base model is *over-confident* (large
  deep nets memorising labels); harmless-to-slightly-worse when it is
  already under-confident.
- **Smoother decision boundary** — modest adversarial robustness bonus.

## When to use

- **Classification with many classes and messy labels** — the standard
  ImageNet recipe uses `ε = 0.1`.
- **Sequence models** (Transformer, GPT-2 pre-training) use `ε = 0.1` by
  default.
- **Anywhere calibration matters at deployment** — pair with
  temperature scaling for a fully post-hoc-calibrated ensemble.

## When NOT to use

- **Downstream distillation** (Müller-Kornblith-Hinton 2019) — label
  smoothing collapses intra-class feature variance and hurts the student.
- **Very few classes** (K = 2, 3) — the smoothed target has a large
  uniform mass that can distort learning.
- **Already-calibrated small models** — the demo here shows it can
  slightly *raise* ECE on a linear classifier that starts under-confident.

## Files

- `python/label_smoothing.py` — from-scratch softmax classifier with
  a soft-target CE loss. Sweep over `ε ∈ {0, 0.05, 0.10, 0.20}` on a
  synthetic 3-class problem with 20 % label noise; report accuracy,
  ECE, NLL, and mean confidence.
- `r/label_smoothing.R` — Keras / torch (R + Python); the loss
  argument is a one-liner in every modern framework.

## Assumptions & caveats

- **ε is problem-dependent** — 0.05 – 0.10 is standard for classification;
  higher for very noisy label distributions.
- **Interacts with weight decay** — a very heavy weight-decay may
  double-count the confidence penalty; tune together.
- **Non-uniform smoothing** — some papers use `y_smooth = (1−ε)·y +
  ε·prior(y)` with a class-frequency prior instead of Uniform.
- **KL(y‖p) ≠ KL(p‖y)** — CE uses the former; if you swap direction,
  gradients look different.

## Related in this repo

- `calibration-scaling` — post-hoc calibration (Platt, isotonic,
  temperature) that stacks with label smoothing.
- `mixup`, `cutmix` — data-side smoothing rather than label-side.
- `deep-mlp-backprop`, `dropout-batchnorm` — the training loop that
  hosts the loss.
- `knowledge-distillation` — the *place* where label smoothing hurts;
  see Müller 2019.

## Run

```
python techniques/label-smoothing/python/label_smoothing.py
Rscript techniques/label-smoothing/r/label_smoothing.R
```

**Refs:** Szegedy, C. et al. "Rethinking the Inception architecture for computer vision." *CVPR*, 2016; Müller, R., Kornblith, S. & Hinton, G. "When does label smoothing help?" *NeurIPS*, 2019.

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

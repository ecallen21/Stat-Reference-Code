# Mixup (Reference Ch 30 Robustness)

**Train on convex combinations of training pairs** — Zhang, Cissé,
Dauphin & Lopez-Paz (2018). One of the cheapest, most transferable
regularisers ever proposed.

## Formula

For each mini-batch:

```
λ  ~  Beta(α, α)
x̃  =  λ · x_i  +  (1 − λ) · x_j
ỹ  =  λ · y_i  +  (1 − λ) · y_j          (one-hot mixed)
```

Cross-entropy loss on `(x̃, ỹ)` replaces the vanilla ERM step.

## Behaviour

- `α → 0` — `λ` concentrates near 0 or 1 → almost no mixing (vanilla).
- `α = 0.2 – 0.4` — image-classification default.
- `α → ∞` — `λ → 0.5` → every sample is a 50/50 mix, usually too aggressive.

## Why it helps

- **Vicinal risk minimisation** — the model learns behavior *between*
  training points, not just at them.
- **Regularises confidence** — like label smoothing but data-dependent.
- **Robustness bonus** — small improvements against `L_p` adversarial
  attacks, label noise, and OOD inputs.
- **Composable** with CutMix, RandAugment, AugMix.

## When to use

- **Image classification** — a default component of ResNet, EfficientNet,
  ViT recipes.
- **Tabular / audio / graph** — reduced-effect but usually still positive.
- **Small training sets with noisy labels** — the vicinal points break
  the memorisation of noise.

## When NOT to use

- **Regression with structured targets** — the mixed target loses
  physical meaning.
- **Metric learning / contrastive** — mixing anchors and positives
  distorts the distance space.
- **Very small K** — with `K = 2` and heavy mixing the classifier
  spends most of its capacity fitting the uniform region.

## Files

- `python/mixup.py` — from-scratch: `Beta(α, α)` mixing of every batch;
  softmax classifier trained with mixed one-hot targets. Sweep vanilla
  vs mixup on synthetic 3-class data with 15 % label noise; report clean
  accuracy, ECE, mean confidence, and robustness to Gaussian input noise.
- `r/mixup.R` — Keras / torch (R + Python); the mixing loop is a
  few lines in every framework.

## Assumptions & caveats

- **α is problem-dependent** — grid-search over {0.1, 0.2, 0.4, 1.0}.
- **Mixing across dissimilar classes** can be harmful — some variants
  (Manifold Mixup, Guo 2019) mix hidden representations instead.
- **Interaction with BatchNorm** — mix within each BN mini-batch, not
  across.
- **Not a substitute for calibration** — pair with
  `calibration-scaling` for well-calibrated deployment probabilities.
- **Not a defence certificate** — for a *proof* of robustness use
  `randomized-smoothing`.

## Related in this repo

- `cutmix` — patch-swap alternative that keeps local statistics intact.
- `label-smoothing` — label-side smoothing (no data mixing).
- `fgsm-adversarial`, `pgd-adversarial-training`, `trades-adversarial`
  — dedicated adversarial defences.
- `dropout-batchnorm`, `deep-mlp-backprop` — the underlying training loop.

## Run

```
python techniques/mixup/python/mixup.py
Rscript techniques/mixup/r/mixup.R
```

**Refs:** Zhang, H., Cissé, M., Dauphin, Y. & Lopez-Paz, D. "mixup: beyond empirical risk minimisation." *ICLR*, 2018; Guo, H., Mao, Y. & Zhang, R. "MixUp as locally linear out-of-manifold regularisation." *AAAI*, 2019.

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

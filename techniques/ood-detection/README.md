# Out-of-Distribution Detection (Reference Ch 29 UQ)

Score every test input by how *inlier* it looks under the training
distribution, and threshold to flag OOD. Cheap, model-agnostic baselines
often beat elaborate methods (Hendrycks-Gimpel 2017).

## Three canonical scores

**MSP — Maximum Softmax Probability** (Hendrycks-Gimpel 2017)

```
score(x) = max_k p̂_k(x)         (higher = more inlier)
```

**Energy** (Liu-Zhang-Owens-Li 2020)

```
E(x) = −T · log Σ_k exp(z_k(x) / T)
score(x) = −E(x)                  (higher = more inlier)
```

Numerically stabler and often much better than MSP without any fine-tuning.

**Mahalanobis** (Lee-Lee-Lee-Shin 2018)

```
μ_k    = class-k mean of penultimate features
Σ      = pooled class-conditional covariance
score(x) = − min_k (φ(x) − μ_k)ᵀ Σ⁻¹ (φ(x) − μ_k)
```

Assumes penultimate features are approximately Gaussian per class; when
that holds, it is often the strongest zero-training-cost method.

## Evaluation

- **AUROC(in-vs-out)** — treat in-distribution as positive; higher = better.
- **FPR@95%TPR** — fraction of OOD kept as in-distribution when accepting
  95 % of the true in-distribution inputs; lower = better.

## When to use

- **Any deployed classifier** — MSP is one line.
- **Deep features available** — Mahalanobis or Energy on the penultimate
  activations.
- **Safety-critical decisions** — pair with `selective-prediction` so the
  model can abstain.

## Files

- `python/ood_detection.py` — from-scratch MSP + Energy + Mahalanobis on
  synthetic in-distribution Gaussian clusters vs uniform far-field OOD.
  Demo: **Energy AUROC = 1.000, Mahalanobis AUROC = 1.000, MSP AUROC =
  0.447** (MSP fails when logits saturate).
- `r/ood_detection.R` — `reticulate` + `pytorch-ood` / `openood`; native
  R via `mvoutlier` / `isotree` / `robustbase::covMcd` for tabular data.

## Assumptions & caveats

- **Deep-net logits saturate** — MSP is a poor score on high-capacity
  networks even when the classifier is well trained.
- **Temperature scaling** the softmax first often improves MSP and helps
  Energy (used implicitly).
- **Mahalanobis** requires representative class-conditional Gaussians;
  ties to `linear-discriminant-analysis` intuition.
- **Deep-ensemble variance** or `mc-dropout` sample variance are strong
  ensemble-based OOD signals — combine when you can afford it.
- **Near-OOD is hard** — semantically similar OOD (rotated MNIST, blurred
  ImageNet) requires stronger methods (ViM, KNN, SNGP).
- **No calibration guarantee** — an OOD score is a ranking, not a
  probability. Use `conformal-classification` if you need coverage.

## Related in this repo

- `mc-dropout`, `deep-ensembles`, `bayesian-neural-network`,
  `evidential-deep-learning` — model-uncertainty scores also work as
  OOD signals.
- `selective-prediction` — the natural user of an OOD score.
- `mahalanobis-distance` (if present) / `linear-discriminant-analysis` —
  the Gaussian assumption used here.
- `isolation-forest`, `dbscan` — classical density-based OOD.

## Run

```
python techniques/ood-detection/python/ood_detection.py
Rscript techniques/ood-detection/r/ood_detection.R
```

**Refs:** Hendrycks, D. & Gimpel, K. "A baseline for detecting misclassified and out-of-distribution examples in neural networks." *ICLR*, 2017; Liu, W., Zhang, X., Owens, J.D. & Li, Y. "Energy-based out-of-distribution detection." *NeurIPS*, 2020; Lee, K., Lee, K., Lee, H. & Shin, J. "A simple unified framework for detecting out-of-distribution samples (Mahalanobis)." *NeurIPS*, 2018.

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

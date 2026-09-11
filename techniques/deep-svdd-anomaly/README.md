# Deep SVDD — Deep One-Class Classification (Reference §47.240)

Ruff et al. (2018, ICML). Learn a neural network φ_θ that maps
normal training data into a compact hypersphere of centre c in
the embedding space:

    L = (1/n) Σ_i ‖φ_θ(x_i) − c‖²  +  λ ‖θ‖²

Anomaly score at test time = distance from c:

    s(x) = ‖φ_θ(x) − c‖

Unsupervised; only NORMAL data at training time. Related to
OC-SVM but with learned features.

## Files

- `python/deep_svdd_anomaly.py` — toy linear encoder + centre
  fit on 200 normal + 30 anomaly samples in R⁴:
  - Threshold at normal-95 %ile: **TP 26 / 30 anomalies**, FP
    10 / 200 normals.
- `r/deep_svdd_anomaly.R` — no R port; recommends
  `pyod.models.DeepSVDD`.

## When to use

- **One-class / novelty detection** with labelled NORMAL data
  only.
- **When learned features** matter more than distance metric
  choice.
- **High-dim data** where OC-SVM is prohibitive.

## When NOT to use

- **When labeled anomalies exist** — supervised classification
  cheaper.
- **Very small normal sets** — Deep SVDD collapses to trivial
  solution (must set c ≠ 0 and remove bias terms).

## Assumptions & caveats

- **Centre c** — set to mean of initial embeddings; keep FIXED
  during training.
- **No bias terms / no biased activations** — otherwise trivial
  minimum with φ = c.
- **Weight decay** essential to avoid collapse.
- **Contamination in training data** hurts; pre-filter or use
  soft-boundary variants.

## Related in this repo

- `isolation-forest-anomaly`, `loda-anomaly-detection` —
  sibling anomaly detectors.
- `variational-autoencoder`, `autoencoder` — reconstruction-
  based novelty cousins.
- `ood-detection` — related distribution-shift neighbour.

## Run

```
python techniques/deep-svdd-anomaly/python/deep_svdd_anomaly.py
Rscript techniques/deep-svdd-anomaly/r/deep_svdd_anomaly.R
```

**Refs:** Ruff, L. et al. "Deep one-class classification (Deep SVDD)." *ICML*, 2018.

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

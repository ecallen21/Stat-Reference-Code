# SimCLR — Contrastive Learning of Visual Representations (§47.153)

Chen, Kornblith, Norouzi & Hinton (2020). Two random augmentations
of each image form a **positive pair**; every other image in the
batch is a **negative**. NT-Xent loss:

    L_ij = −log( exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ) )

with z = g(f(x)), projection head g. Learns representations without
labels; a linear classifier on top nearly matches supervised
training on ImageNet.

## Files

- `python/simclr_contrastive.py` — from-scratch linear-encoder
  SimCLR with SPSA gradient estimate on the NT-Xent objective.
  UCI digits (64-d):
  - LR on raw pixels (supervised upper bound): 0.961
  - LR on random 16-D projection (no learning): 0.794
  - LR on **SimCLR 16-D features (SSL, no labels)**: **0.883**
    — closes ~48 % of the gap between random projection and
    supervised.
- `r/simclr_contrastive.R` — recommends `lightly`, `solo-learn`,
  `pytorch-metric-learning.NTXentLoss` (all Python).

## When to use

- **Label-scarce** image / audio / text representation learning.
- **Pretraining** for downstream fine-tuning (linear probe,
  fine-tune).
- **Rich augmentation pipelines** are available (colour jitter,
  crop, blur for images).

## When NOT to use

- **Fully labelled** downstream task with enough data — plain
  supervised training wins.
- **Small batch sizes** — SimCLR needs many negatives per
  batch (paper uses 4096-8192).
- **Domains with weak augmentation invariances** (e.g. sequential
  logs) — negatives become uninformative.

## Assumptions & caveats

- **Temperature τ** ~ 0.1-0.5 critical; too small = collapse.
- **Projection head** g is discarded at test time; the encoder f
  is what's transferred.
- **Batch size scaling** — larger batches → more negatives → better
  features. Momentum-encoder MoCo relaxes this.
- **Augmentation choice** dominates final quality (strong crop +
  colour distortion for images).

## Related in this repo

- `contrastive-learning`, `contrastive-predictive-coding` —
  general contrastive-SSL framework.
- `barlow-twins`, `byol-simsiam` — non-contrastive SSL siblings.
- `siamese-networks`, `deep-metric-learning-triplet` — supervised
  contrastive cousins.

## Run

```
python techniques/simclr-contrastive/python/simclr_contrastive.py
Rscript techniques/simclr-contrastive/r/simclr_contrastive.R
```

**Refs:** Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. "A simple framework for contrastive learning of visual representations." *ICML*, 2020; Chen, X., Fan, H., Girshick, R. & He, K. "Improved baselines with momentum contrastive learning." *arXiv:2003.04297*, 2020.

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

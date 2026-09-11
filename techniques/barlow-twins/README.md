# Barlow Twins (Reference §47.154)

Zbontar, Jing, Misra, LeCun & Deny (2021). Two augmentations of
each input are encoded and projected; the cross-correlation matrix
C between the projections is pushed toward the identity:

    L = Σ_i (1 − C_ii)²   +   λ · Σ_{i≠j} C_ij²
        (invariance term)      (redundancy-reduction term)

Uses NO negative pairs and NO memory bank — simpler than
SimCLR / MoCo — yet competitive on ImageNet linear-probe.

## Files

- `python/barlow_twins.py` — from-scratch linear projection with
  SPSA gradient on the Barlow objective. UCI digits (64-d):
  - LR on raw pixels: 0.961
  - LR on random 16-D projection: 0.794
  - **LR on Barlow-Twins 16-D (SSL)**: **0.828**
  - Cross-corr diag mean = 0.45 (pushed toward 1),
    off-diag |mean| = 0.10 (pushed toward 0).
- `r/barlow_twins.R` — recommends `lightly`, `solo-learn`,
  Facebook Research reference (all Python).

## When to use

- **Label-scarce** representation learning without needing large
  batch sizes or memory banks.
- **Small-batch training** (unlike SimCLR / MoCo).
- **Simple SSL baseline** to start any new representation project.

## When NOT to use

- **Very small embedding dim** — off-diagonal penalty needs D >
  ~64 to shine.
- **Fully labelled** downstream with abundant data — supervised
  wins.
- **Domains with weak augmentation invariances**.

## Assumptions & caveats

- **λ** weights redundancy term; paper uses 5e-3, works over 1e-4
  → 1e-2.
- **Batch normalisation** across the batch is essential (that's
  where cross-correlation comes from).
- **Embedding dim D** — larger is better (paper uses 8192 on
  ImageNet); off-diagonal grows as D², so scale λ ~ 1/D².
- **No collapse safeguard needed** — the redundancy term prevents
  trivial constant embeddings.

## Related in this repo

- `simclr-contrastive`, `contrastive-learning`,
  `byol-simsiam` — SSL siblings.
- `deep-metric-learning-triplet`, `siamese-networks` —
  supervised metric-learning cousins.
- `contrastive-predictive-coding` — an earlier positive-pair SSL.

## Run

```
python techniques/barlow-twins/python/barlow_twins.py
Rscript techniques/barlow-twins/r/barlow_twins.R
```

**Refs:** Zbontar, J., Jing, L., Misra, I., LeCun, Y. & Deny, S. "Barlow Twins: Self-supervised learning via redundancy reduction." *ICML*, 2021; Bardes, A., Ponce, J. & LeCun, Y. "VICReg: Variance-invariance-covariance regularization for self-supervised learning." *ICLR*, 2022.

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

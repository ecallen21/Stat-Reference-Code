# Tomek Links Undersampling (Reference §47.249)

Tomek (1976). A pair `(x_i, x_j)` forms a TOMEK LINK if they
belong to different classes AND each is the other's nearest
neighbour. Removing the majority-class member of each Tomek
link CLEANS the class boundary.

Often paired with SMOTE (SMOTE-Tomek): first oversample,
then clean.

## Files

- `python/tomek_links_undersampling.py` — Numpy + sklearn NN.
  Demo: n=2000, 20 features, 90/10 imbalance, 10% label flip.
  Removed 21 Tomek-linked majority samples; LR precision holds
  at 0.97 while recall rises from 0.36 → 0.39, F1 0.53 → 0.55
  — small but principled gain from a cleaner boundary.
- `r/tomek_links_undersampling.R` — `unbalanced::ubTomek`,
  `UBL::TomekClassif`, `themis::step_tomek` (R);
  `imbalanced-learn.TomekLinks` / `SMOTETomek`, from-scratch
  (Python).

## When to use

- **Boundary cleaning** after oversampling (SMOTE-Tomek).
- **Mildly imbalanced** classification where removing boundary
  noise matters more than balancing.
- **As a preprocessing step** for k-NN classifiers whose
  performance suffers under class overlap.

## When NOT to use

- **Severe imbalance** where undersampling alone loses too
  much majority signal — combine with resampling instead.
- **Small datasets** — Tomek removals may drop critical
  boundary information.
- **Noise-dominated data** — removals will chase noise; use
  ENN (majority-vote cleaning) for robustness.

## Assumptions & caveats

- **Symmetric NN required** — the pair must be MUTUAL nearest
  neighbours; asymmetric variants (one-sided selection) exist.
- **Distance metric matters** — standardise features first;
  Manhattan / Mahalanobis change which pairs qualify.
- **Fold isolation** — compute Tomek links inside cross-
  validation to avoid leakage.
- **Removals are limited** by mutual-NN structure; only a
  fraction of majority is ever eligible.

## Related in this repo

- `edited-nn-cleaning` — the majority-vote k-NN cleaning
  companion.
- `smote-oversampling`, `adasyn-oversampling` — often combined
  with Tomek for SMOTE-Tomek / ADASYN-Tomek.
- `class-imbalance` — the overall framing.
- `matthews-correlation-coefficient` — recommended metric.

## Run

```
python techniques/tomek-links-undersampling/python/tomek_links_undersampling.py
Rscript techniques/tomek-links-undersampling/r/tomek_links_undersampling.R
```

**Refs:** Tomek, I. "Two Modifications of CNN." *IEEE Trans. Syst. Man Cybern.*, SMC-6(11): 769-772, 1976.

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

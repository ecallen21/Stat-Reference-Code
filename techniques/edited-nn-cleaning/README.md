# Edited Nearest Neighbours (Reference §47.250)

Wilson (1972). Remove any sample `x` whose class label
disagrees with the MAJORITY vote of its `k` nearest neighbours:

```
for each x in dataset:
    if class(x) != mode(class(k-NN of x)): remove x
```

Cleans mislabelled or ambiguous samples. Used in SMOTE-ENN
pipelines: oversample then ENN-clean.

## Files

- `python/edited_nn_cleaning.py` — Numpy + sklearn NN. Demo:
  n=2000, 20 features, 90/10 imbalance, 15% label flip. ENN
  removed 200 samples (14.3%), split maj=45 / min=155 — the
  algorithm removed more minority than majority because
  flipped minorities look like majorities to their neighbours.
  LR precision rises 0.94 → 0.97 with recall drop 0.35 → 0.32
  — illustrates ENN's tendency to shrink the minority.
- `r/edited_nn_cleaning.R` — `unbalanced::ubENN`,
  `UBL::ENNClassif`, `themis` (R);
  `imbalanced-learn.EditedNearestNeighbours` / `AllKNN` /
  `SMOTEENN`, from-scratch (Python).

## When to use

- **Noisy labels** where boundary confusion dominates class
  overlap — ENN removes label-flipped points.
- **After oversampling** (SMOTE-ENN) — cleans synths that
  landed in majority territory.
- **k-NN classifiers** — Wilson's original motivation was to
  clean training data for edited-NN classification.

## When NOT to use

- **Severe minority shortage** — ENN can decimate small
  minority classes; combine with SMOTE first.
- **Deliberately overlapping classes** (e.g. calibration
  studies) — the "cleaned" boundary loses the overlap you
  wanted to model.
- **Very high dimension** — NN votes become unreliable.

## Assumptions & caveats

- **Odd `k`** avoids vote ties (typical: k=3, 5).
- **Standardise features** — distance-based method.
- **Repeated ENN (AllKNN)** iterates the cleaning; more
  aggressive but risks over-shrinking.
- **Fold isolation** — apply within cross-validation folds.
- **Class-wise effects**: minority is at higher removal risk
  in imbalanced settings — check by-class removal counts.

## Related in this repo

- `tomek-links-undersampling` — pair-based boundary cleaning
  companion.
- `smote-oversampling` — SMOTE-ENN combines both.
- `adasyn-oversampling` — similar oversampling partner.
- `class-imbalance` — the overall framing.

## Run

```
python techniques/edited-nn-cleaning/python/edited_nn_cleaning.py
Rscript techniques/edited-nn-cleaning/r/edited_nn_cleaning.R
```

**Refs:** Wilson, D.L. "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data." *IEEE Trans. Syst. Man Cybern.*, SMC-2(3): 408-421, 1972.

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

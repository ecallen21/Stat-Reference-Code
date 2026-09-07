# Semi-Supervised Pseudo-Labelling (Reference §47.39)

Lee (2013). Iteratively augment a small labelled set with
high-confidence predictions on unlabelled data:

1. Train the classifier on labelled data `L`.
2. Predict on unlabelled data `U`.
3. Add high-confidence predictions (`max prob > τ`) as pseudo-
   labels.
4. Retrain.

Related: Yarowsky (1995) self-training, Blum-Mitchell (1998)
co-training, FixMatch (Sohn 2020) with weak+strong augmentation.

## Files

- `python/semi_supervised_pseudo_labeling.py` — pseudo-label
  self-training loop for a logistic classifier from scratch.
  Demo (n_labelled=30, n_unlabelled=500, τ=0.9): labelled-only
  baseline accuracy 0.86; after 5 rounds of pseudo-labelling
  ~0.85. Illustrates the classic confirmation-bias caveat — help
  or harm depends on baseline calibration.
- `r/semi_supervised_pseudo_labeling.R` — `RSSL`, `mlr3verse`,
  `SSL` (R);
  `sklearn.semi_supervised.SelfTrainingClassifier / LabelPropagation`,
  scikit-uplift / snorkel (Python).

## When to use

- **Abundant unlabelled data, expensive labels** — text
  classification, image tagging, medical annotation.
- **Well-calibrated base model** — high-confidence subset
  reliably correct.
- **Bootstrap for cold-start** — get a working classifier out of
  scarce labels.

## When NOT to use

- **Overconfident / miscalibrated base model** — confirmation bias
  reinforces errors.
- **Distribution shift** — pseudo-labels drift; use consistency
  regularisation (FixMatch) or entropy minimisation.
- **Strong labelled baseline** — marginal gains, high risk of
  degradation.

## Assumptions & caveats

- **Threshold `τ`** — 0.9 typical; too low → noisy labels, too
  high → few labels added.
- **Class balance** — pseudo-labelling can amplify majority class;
  balance per-class caps.
- **Confidence ≠ correctness** — combine with entropy min /
  consistency regularisation (MixMatch, FixMatch).
- **Iterate to convergence** or fixed-K rounds; monitor validation
  accuracy at each step.

## Related in this repo

- `active-learning-query-strategies` — orthogonal label-acquisition.
- `covariate-shift-adaptation`, `concept-drift-adwin` — data-shift
  cousins.
- `contrastive-learning`, `jepa-self-supervised` — SSL siblings.

## Run

```
python techniques/semi-supervised-pseudo-labeling/python/semi_supervised_pseudo_labeling.py
Rscript techniques/semi-supervised-pseudo-labeling/r/semi_supervised_pseudo_labeling.R
```

**Refs:** Lee, D.-H. "Pseudo-label: the simple and efficient semi-supervised learning method for deep neural networks." *ICML Workshop*, 2013; Sohn, K. et al. "FixMatch: simplifying semi-supervised learning with consistency and confidence." *NeurIPS*, 2020.

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

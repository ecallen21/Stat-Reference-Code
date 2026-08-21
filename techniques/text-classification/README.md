# Text Classification (Reference §25.6)

Two workhorse baselines for supervised text classification.

## Multinomial Naive Bayes

```
P(y = c | x) ∝ P(y = c) · Π_w P(w | c)^{x_w}
P(w | c) = (count(w, c) + α) / (Σ_v count(v, c) + α · |V|)      (Laplace smoothing)
```

- Very fast to fit and predict; competitive with much fancier models on tf-based inputs.
- **Complement NB** (Rennie et al. 2003) fixes NB's bias toward majority classes on skewed data.

## TF-IDF + regularised logistic regression

- Multinomial logistic on TF-IDF features; L1 or L2 regularisation.
- Reliably beats NB when features overlap between classes (real corpora), especially with character n-grams.
- **Linear SVM** on TF-IDF is a close cousin and often preferred for very large sparse feature spaces.

## Evaluation

- **Per-class** precision, recall, F1 (macro-average for balanced view, micro/weighted for imbalanced).
- **Confusion matrix** for error analysis.
- **PR-AUC** rather than ROC-AUC when the positive class is rare.
- **k-fold CV** for small corpora; time-based holdouts for streaming.

## When to use

- **Baseline** for any new text-classification problem — always try MultinomialNB + LR on TF-IDF before reaching for BERT.
- **High-throughput / low-latency** classification — NB and LR fit in seconds and serve in microseconds.
- **Interpretability** — LR coefficients directly tell you which words matter.
- **Modern option**: fine-tune a transformer (DistilBERT, RoBERTa) or use a zero-shot classifier for topic-labelling with no labels.

## Files

- `python/text_classification.py` — from-scratch multinomial NB + multiclass logistic (softmax + L2). Demo (D=120 docs from 3 disjoint topical vocabs, 90/30 train/test split): NB and LR both hit 100% test accuracy; sklearn `MultinomialNB` and `LogisticRegression` on the same TF-IDF match.
- `r/text_classification.R` — `quanteda.textmodels::textmodel_nb / svm`, `glmnet::cv.glmnet(family='multinomial')`, `text2vec` pipelines.

## Assumptions & caveats

- **NB independence assumption** is violated by every real text feature set; NB still works well because it only needs the right *ranking* of class probabilities.
- **Class imbalance** — accuracy misleads; report macro F1 and per-class metrics.
- **Feature engineering matters more than the classifier** — char n-grams, TF-IDF variants, stopword tuning, min_df/max_df.
- **Regularisation for LR** — L1 for sparse feature selection; L2 for smoothness; ElasticNet for the mix. Cross-validate the strength.
- **Calibration** — NB probabilities are notoriously mis-calibrated (see `calibration-scaling`); use Platt scaling / isotonic on the held-out set.
- **Modern practice** — fine-tuning a pretrained transformer beats NB / LR + TF-IDF by 5–15 F1 on most benchmarks, at 100–1000× compute cost. Choose based on latency budget.

## Related in this repo

- `text-preprocessing`, `tfidf-bm25`, `word-embeddings` — input pipelines.
- `naive-bayes` — the general classifier.
- `ridge-lasso-elasticnet`, `logistic-regression` — general regularised regression.
- `svm-classifier` — linear-SVM sibling of TF-IDF logistic.
- `class-imbalance`, `calibration-scaling` — critical companion techniques for real corpora.

## Run

```
python techniques/text-classification/python/text_classification.py
Rscript techniques/text-classification/r/text_classification.R
```

**Refs:** McCallum, A. & Nigam, K. "A comparison of event models for Naive Bayes text classification." *AAAI Workshop*, 1998; Rennie, J.D.M. et al. "Tackling the poor assumptions of Naive Bayes text classifiers." *ICML*, 2003; Joachims, T. "Text categorization with Support Vector Machines: learning with many relevant features." *ECML*, 1998.

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

# Naive Bayes (Reference §26.10)

Classifier applying Bayes' rule under a **conditional independence** assumption:

```
Pr(y = c | x) ∝ Pr(y = c) · Π_j Pr(x_j | y = c)
```

The "naive" bit is treating features as independent given the class. Wildly wrong in general, but works surprisingly well for text and sparse categorical data — decision boundaries are often correct even when the individual probability estimates are miscalibrated.

## Variants

- **Gaussian NB** — continuous features: `Pr(x_j | y = c) = N(μ_jc, σ_jc²)`.
- **Multinomial NB** — count features (bag-of-words):

```
Pr(w_j | y = c) = (count_jc + α) / (Σ_j count_jc + α V)      Laplace smoothing α > 0
```

- **Bernoulli NB** — binary features (word present / absent).
- **Complement NB** — better on imbalanced text corpora.

## Prediction

Pick `c` maximizing `log Pr(y = c) + Σ_j log Pr(x_j | y = c)`.

## Files

- `python/naive_bayes.py` — from-scratch Gaussian and Multinomial NB with Laplace smoothing. Demo: 3-blob Gaussian → 96.0% accuracy (sklearn GaussianNB 95.7%); synthetic bag-of-words → 100% on well-separated topics.
- `r/naive_bayes.R` — `e1071::naiveBayes` or `naivebayes::naive_bayes`.

## When to use

- **Text classification** (spam, sentiment, topic) — Multinomial NB is a classic strong baseline.
- **Very fast training + prediction** — closed-form MLE, no gradient descent.
- **Small samples** — the strong independence assumption regularizes.
- **Streaming / online** updates — trivially incremental.

## When NOT to use

- **Correlated features** — Naive Bayes probability estimates become badly miscalibrated (though the argmax class often still stays right).
- **Calibrated probabilities required** — Platt scaling or isotonic calibration after NB.
- **Feature interactions** matter for the decision boundary — trees / GBM / SVM are better.

## Assumptions & caveats

- **Zero-frequency**: any `Pr(x_j | c) = 0` zeroes the whole product. **Laplace smoothing** (`α ≥ 1`) is essential.
- **Log-space computation** — avoid numerical underflow on many-feature products.
- **Non-negative features only** for Multinomial NB; use TF-IDF or raw counts.

## Run

```
python techniques/naive-bayes/python/naive_bayes.py
Rscript techniques/naive-bayes/r/naive_bayes.R
```

**Refs:** McCallum, A. & Nigam, K. "A comparison of event models for naive Bayes text classification." *AAAI-98 Workshop*, 1998; Rennie, J.D.M. et al. "Tackling the poor assumptions of naive Bayes text classifiers." *ICML*, 2003.

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

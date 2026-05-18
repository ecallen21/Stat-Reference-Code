# Multinomial Logistic Regression (Reference §7.4)

For an **unordered** categorical outcome `Y ∈ {1, …, K}`, pick a reference class (say `K`) and model the `K−1` log-odds against it:

`log P(Y = k | x) / P(Y = K | x) = x'β_k`,   `k = 1, …, K−1`

Equivalently, `P(Y = k | x) = softmax( x'β₁, …, x'β_{K−1}, 0 )_k`.

`exp(β_kj)` is the odds ratio for class `k` vs. the reference for a unit increase in `xⱼ`.

## Fitting

Direct MLE via BFGS on the (K−1)·p flattened parameter vector. Spark MLlib's `LogisticRegression(family="multinomial")` uses L-BFGS on the same objective.

## Vs. ordinal logistic

Use **multinomial** when categories are genuinely unordered (color preference, diagnosis category). Use **ordinal logistic** (`techniques/ordinal-logistic`) when there's a clear ordering (Likert, severity stage) — it uses far fewer parameters (`(K−1) + p` vs. `(K−1)·p`) and is more powerful when the proportional-odds assumption holds.

## Reference-class conventions

Different libraries pick different reference classes:
- This implementation: **class K (the largest label)** is the reference.
- `nnet::multinom` in R: **class 1** (first level) is the reference.
- `statsmodels.MNLogit`: **class 0/1** (first level) is the reference.
- Spark MLlib `LogisticRegression(family="multinomial")`: returns coefficients for **all K** classes (no implicit reference); they're estimable up to a constant.

The fitted probabilities are identical across conventions; the displayed coefficient values look different by a constant offset per row.

## Files
- `python/multinomial_logistic.py` — from-scratch softmax MLE via BFGS; matches `statsmodels.MNLogit` up to reference-class choice.
- `r/multinomial_logistic.R` — from-scratch + `nnet::multinom`.
- `pyspark/multinomial_logistic.py` — `pyspark.ml.classification.LogisticRegression(family="multinomial")`.

## Run
```
python techniques/multinomial-logistic/python/multinomial_logistic.py
Rscript techniques/multinomial-logistic/r/multinomial_logistic.R
python techniques/multinomial-logistic/pyspark/multinomial_logistic.py
```

**Refs:** Agresti, *Categorical Data Analysis*, 3rd ed., Wiley, 2013; Hosmer, Lemeshow & Sturdivant, *Applied Logistic Regression*, 3rd ed., 2013 (Ch. 8).

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

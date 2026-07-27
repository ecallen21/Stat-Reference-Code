# Principal Component Analysis (Reference §9.3)

PCA finds an orthogonal rotation of the `p` variables into new axes (**principal components**) ordered by the variance they explain. The first PC is the direction of maximum variance in the data; each subsequent PC is orthogonal to all previous ones and captures the maximum remaining variance.

## Algorithm (SVD form)

```
X (n × p) centered (and optionally scaled) → Xc
Xc / √(n − 1)  =  U Σ V'                    (SVD)

Loadings V     : p × k matrix; columns are the new axes' directions in variable space
Scores  U·Σ·√(n−1) = Xc·V  : n × k projection of the data into PC space
Variances Σ²  : per-PC eigenvalues (variance explained on each axis)
```

Equivalent to eigen-decomposing the sample covariance `Xc' Xc / (n − 1)`, but SVD is numerically stabler and doesn't require forming the p×p covariance matrix.

## Options

- **Covariance PCA** (default): PCA on the raw variables. Use when variables share a scale/unit.
- **Correlation PCA** (`scale=True`): divide each column by its SD first. Use when variables have very different scales; otherwise the largest-variance variable dominates.

## Files

- `python/pca.py` — from-scratch SVD-based PCA; loadings, scores, singular values, per-PC and cumulative variance ratios, biplot coordinates. Explained-variance ratios match `sklearn.decomposition.PCA` to 12 dp.
- `r/pca.R` — from-scratch + base `stats::prcomp`.
- `pyspark/pca.py` — MLlib `pyspark.ml.feature.PCA` on distributed feature vectors; explicit mean-centering (MLlib PCA does not center by default).

## Assumptions

- No probability model required — PCA is descriptive geometry, not inference.
- Linear relationships; nonlinear structure needs kernel PCA or manifold methods.
- Scaling matters — always inspect variable scales before deciding covariance vs. correlation PCA.

## Run

```
python techniques/pca/python/pca.py
Rscript techniques/pca/r/pca.R
python techniques/pca/pyspark/pca.py
```

**Refs:** Hotelling, H. "Analysis of a complex of statistical variables into principal components." *J. Educ. Psych.* 24(6/7), 417–441/498–520, 1933; Jolliffe, I.T. *Principal Component Analysis*, 2nd ed., Springer, 2002; Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 14).

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

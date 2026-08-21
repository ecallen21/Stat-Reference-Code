# Partial Least Squares Regression (Reference §5.31)

Regression method (Wold 1966) that constructs latent components maximizing the **covariance** between `X` and `y`. Contrast with PCA regression, which uses X-only variance and can miss `y`-relevant directions.

Especially useful when:

- `p >> n` (many correlated predictors — chemometrics, genomics, sensor arrays).
- `X` columns are highly collinear.
- Multi-output regression (`Y` is multivariate → PLS2).

## NIPALS algorithm (PLS1, single response `y`)

For each component `h = 1, ..., H`:

```
w = Xᵀ y / ‖Xᵀ y‖              weight vector
t = X w                         latent score
p = Xᵀ t / (tᵀ t)               X loading
b = tᵀ y / (tᵀ t)               regression coefficient
X ← X − t pᵀ                     deflate
y ← y − b t
```

**Prediction**: `β̂ = W (Pᵀ W)⁻¹ [b_1, ..., b_H]`.

**Component count** `H` chosen by CV.

## Files

- `python/partial_least_squares.py` — from-scratch NIPALS PLS1 + 5-fold CV to select H. Demo (n = 100, p = 20 with 3-factor latent DGP): CV picks H = 2; in-sample RMSE 0.51 matches `sklearn.cross_decomposition.PLSRegression` closely.
- `r/partial_least_squares.R` — `pls::plsr(y ~ X, ncomp, validation = "CV")` (Mevik canonical package).

## When to use

- **Spectroscopy / chemometrics** — many wavelengths, few samples.
- **Genomics** — thousands of genes, dozens of subjects.
- **Multi-output prediction** with shared latent structure.

## Contrast

|         | PCR                            | PLS                              | Ridge / LASSO                    |
|---------|--------------------------------|----------------------------------|----------------------------------|
| Loading | X-variance (unsupervised)      | X-y covariance (supervised)      | direct coefficient regularization|
| Best    | when y-relevant PCs = top      | fewer components usually         | interpretable coefficients       |

PLS typically needs fewer components than PCR to reach the same accuracy.

## Variants

- **PLS2** — multiple `y` columns share components.
- **SIMPLS** — direct algorithm without deflation (Jong 1993).
- **Sparse PLS** — L1 penalty on the weights; `spls` in R.
- **PLS-DA** — PLS for classification (categorical `y`).

## Assumptions & caveats

- **Center X and y** — done automatically.
- **Standardize X** columns if they have very different scales; recommended in chemometrics.
- **Interpretability**: latent components are linear combinations of `X`, not features themselves; report component loadings and variable-importance-in-projection (VIP).

## Run

```
python techniques/partial-least-squares/python/partial_least_squares.py
Rscript techniques/partial-least-squares/r/partial_least_squares.R
```

**Refs:** Wold, H. "Nonlinear estimation by iterative least squares procedures." In *Research Papers in Statistics*, Wiley, 1966; Höskuldsson, A. "PLS regression methods." *J. Chemometrics* 2(3), 211–228, 1988; Mevik, B.-H. & Wehrens, R. "The pls package: principal component and partial least squares regression in R." *J. Stat. Softw.* 18(2), 1–24, 2007.

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

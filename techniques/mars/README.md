# MARS — Multivariate Adaptive Regression Splines (Reference §5.28)

Adaptive nonparametric regression (Friedman 1991) that grows a piecewise-linear model by searching over all **hinge functions** and their reflections:

```
h_+(x, c) = max(0, x − c)
h_−(x, c) = max(0, c − x)
```

Automatically selects variables, knot locations, and (with `degree > 1`) interactions via products of hinge functions.

## Two-pass algorithm

- **Forward pass**: start with an intercept. At each step add a hinge pair `(h_+(x_j, c), h_−(x_j, c))` that most reduces RSS. Iterate up to `max_terms` or until improvement stalls.
- **Backward pass**: prune terms using **generalized cross-validation**:

```
GCV(M) = RSS(M) / (n · (1 − C(M) / n)²)
C(M)   = M + d (M − 1) / 2      d ≈ 3 penalty per free knot
```

## MARS vs alternatives

|                | Splines / GAM                       | MARS                                 |
|----------------|-------------------------------------|--------------------------------------|
| Basis          | fixed cubic / natural / B-spline    | data-driven hinge functions          |
| Knots          | user-specified                      | discovered                           |
| Interactions   | manual (`s(x1, x2)`)                | automatic (product of hinges)        |
| Smoothness     | smooth (with penalty)               | piecewise-linear (kinks at knots)    |
| Interpretable? | yes (main-effect smooths)           | yes (list of hinges)                 |

## Files

- `python/mars.py` — from-scratch forward pass with variable + knot search (no backward pruning, no interactions). Demo on a piecewise-linear + `|x₂|` target (n = 300, noise sd 0.3): recovers hinges near the true knots (−1, 0.5 on x1) and near 0 on x2; in-sample RMSE 0.06 vs truth.
- `r/mars.R` — `earth::earth(y ~ x1 + x2, degree = 2)` — Milborrow's canonical port with full forward + backward + interactions.

## When to use

- **Nonlinear regression** where you don't know which variables matter or where the breakpoints are.
- **Interactions among many predictors** — MARS with `degree = 2 – 3` finds them automatically.
- **Piecewise-linear** behavior of the response (dose-response, threshold effects).
- **Interpretable ML** — hinge terms are easy to read off.

## When NOT to use

- **High-dimensional (`p > 100`)** — MARS's variable search cost grows fast; consider gradient boosting or LASSO first.
- **Very smooth targets** — GAM with cubic splines fits smoother than MARS's kinks.
- **Very small samples** — MARS overfits; use `nk` (max knots) parsimoniously and cross-validate.

## Assumptions & caveats

- **Additive-hinge structure** — MARS works best when the true function has clear regime changes.
- **Backward pruning + GCV** is essential; without it MARS overfits.
- **Standardize predictors** for cleaner interpretation of knot locations.

## Run

```
python techniques/mars/python/mars.py
Rscript techniques/mars/r/mars.R
```

**Refs:** Friedman, J.H. "Multivariate adaptive regression splines." *Ann. Stat.* 19(1), 1–67, 1991; Milborrow, S. `earth` — Multivariate Adaptive Regression Splines, R package.

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

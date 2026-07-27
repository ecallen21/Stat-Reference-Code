# Wild Bootstrap for Regression with Heteroscedasticity (Reference §10.5)

For a regression `y = Xβ + e` where the error variance depends on `X` (heteroscedasticity), plain-residual bootstrap breaks because it assumes homoscedasticity, and case (row) bootstrap loses efficiency. The **wild bootstrap** keeps each observation's residual but multiplies by a **mean-zero, unit-variance weight**:

```
e*_i  =  ê_i · w_i         with  E[w_i] = 0, Var[w_i] = 1
y*_i  =  X_i β̂ + e*_i
```

Refit on `(X, y*)` to get `β*` and use the empirical distribution over B replicates.

## Weight distributions

| Scheme | Values | Notes |
|---|---|---|
| **Rademacher** | `±1` each w.p. ½ | Fast; symmetric |
| **Mammen** (2-point) | `−(√5−1)/2` w.p. `φ/√5`, `(√5+1)/2` else | Matches E[w] = 0, E[w²] = 1, **E[w³] = 1** — third-moment accuracy |
| **Standard normal** | `w_i ~ N(0, 1)` | Less common; loses discrete-weight advantages |

**Mammen** is recommended in small samples because of the extra third-moment match; **Rademacher** is often used for speed and comparability.

## Why it matters (from the demo)

On synthetic heteroscedastic data (n = 200, `σ_i = 0.5 + |x_1i|`):

| SE(β̂_x1) | Value |
|---|---|
| Naive OLS (homoscedastic) | 0.094 ← wrong |
| Wild bootstrap (Mammen) | 0.135 |
| Wild bootstrap (Rademacher) | 0.136 |
| HC1 robust (sandwich) | 0.136 |

Wild-bootstrap and HC1 agree to 3 dp; both correctly reflect the ~44% larger uncertainty due to heteroscedasticity.

## Files

- `python/wild_bootstrap.py` — Rademacher / Mammen / Normal wild weights; whole-vector or per-coefficient CIs; comparison against homoscedastic OLS SE and HC1 robust SE via `statsmodels`.
- `r/wild_bootstrap.R` — from-scratch + comparison against `lm` (naive) + `sandwich::vcovHC` (HC1).

## Assumptions

- Linear model is correctly specified in its mean.
- Errors are independent (not necessarily identically distributed — that's the point).
- If errors are also correlated (clustered), use a **cluster wild bootstrap** (not implemented here — extension of the same idea with a single weight per cluster).

## Run

```
python techniques/wild-bootstrap/python/wild_bootstrap.py
Rscript techniques/wild-bootstrap/r/wild_bootstrap.R
```

**Refs:** Wu, C.F.J. "Jackknife, bootstrap and other resampling methods in regression analysis." *Ann. Stat.* 14(4), 1261–1295, 1986; Mammen, E. "Bootstrap and wild bootstrap for high-dimensional linear models." *Ann. Stat.* 21(1), 255–285, 1993; Davidson, R. & Flachaire, E. "The wild bootstrap, tamed at last." *J. Econometrics* 146(1), 162–169, 2008.

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

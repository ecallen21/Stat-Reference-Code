# Splines and Segmented Regression (Reference §5.4, §5.22, §5.26, §5.34)

Flexible alternatives to polynomial regression: fit **local** polynomial pieces between chosen **knots** and constrain the pieces to join smoothly. Build the spline **basis** as design-matrix columns, then it's just OLS.

## Why prefer splines over polynomials (§5.34)

- **Local control.** A high-degree polynomial wiggles globally — moving one observation can shift the curve everywhere. Splines are local: a change near `x = 5` doesn't affect the fit near `x = 1`.
- **No Runge phenomenon.** Polynomials oscillate at the boundaries; splines (especially *natural* ones) stay tame.
- **Easy to interpret the flexibility.** Degrees of freedom = number of knots + (basis-specific constants); easy to compare model complexity.

Harrell's *Regression Modeling Strategies* makes the case forcefully: splines are almost always better than polynomials past degree 2.

## What's here

### Piecewise linear (segmented)

`X = [1, x, (x − k₁)₊, (x − k₂)₊, …]`

The slope **changes** by `β_{k+1}` at each knot `kₖ`. Reads as a continuous polyline.

### Natural cubic spline (NCS) — §5.22

`K` knots `k₁ < … < k_K`. Basis (Hastie, Tibshirani & Friedman, ESL §5.4):

```
N₀(x) = 1
N₁(x) = x
N_{j+1}(x) = d_j(x) − d_{K−1}(x)     for j = 1, …, K−2
d_j(x) = ((x − k_j)₊³ − (x − k_K)₊³) / (k_K − k_j)
```

K basis columns total. The "natural" part forces **linear behavior in the tails** — beyond the first and last knot — which is the main reason to prefer NCS over plain B-splines for regression: tails don't blow up.

A common rule: place knots at quantiles of `x` (e.g. `0.1, 0.3, 0.5, 0.7, 0.9` for `K = 5`). With one continuous predictor, `K = 3–5` is plenty; with several, fewer per predictor.

### Breakpoint (unknown knot)

When the location of the slope change is **unknown** and itself estimated. Implemented here as a simple grid search; the canonical algorithm (Muggeo 2003, R's `segmented` package) iteratively re-linearizes around the current estimate and solves for the break point's update.

## Files
- `python/splines_segmented.py` — from-scratch piecewise-linear basis, natural cubic spline basis (ESL recipe), single-breakpoint grid search; compares against `patsy.dmatrix("cr(x, df=...)")`. Demo recovers the true breakpoint (4.0 → estimated ≈ 3.94) and the per-segment slopes (1.0 / −0.5 → 1.02 / −0.50).
- `r/splines_segmented.R` — from-scratch versions + `splines::ns` (natural cubic basis), `splines::bs` (B-spline basis), `segmented::segmented` (canonical breakpoint estimation), `rms::rcs` (restricted cubic spline, Harrell's preferred form).
- PySpark: N/A — once the spline basis columns are built, it's a multiple regression (`techniques/multiple-linear-regression/pyspark/`). The basis itself is a per-row transformation; build it with `pyspark.ml.feature` or a UDF, then hand to `LinearRegression`.

## Run
```
python techniques/splines-segmented/python/splines_segmented.py
Rscript techniques/splines-segmented/r/splines_segmented.R
```

**Refs:** de Boor, *A Practical Guide to Splines*, Springer, 1978; Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009 (Ch. 5); Harrell, *Regression Modeling Strategies*, 2nd ed., 2015 (Ch. 2 — restricted cubic splines); Muggeo, "Estimating Regression Models with Unknown Break-Points," *Stat. in Medicine* 22, 3055–3071, 2003.

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

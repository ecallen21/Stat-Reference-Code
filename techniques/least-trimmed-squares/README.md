# Least Trimmed Squares (Reference §47.373)

Rousseeuw (1984 JASA); Rousseeuw-Van Driessen (2006, FAST-LTS).
Robust regression minimising the sum of the SMALLEST `h`
squared residuals:

```
min_β  Σ_{i=1..h} r_(i)²(β)
```

where `r_(i)` is the i-th order statistic of the squared
residuals. Breakdown point up to 50 % (vs OLS's 0 %) when
`h = ⌈(n + p + 1)/2⌉`. FAST-LTS algorithm:

1. Many random elemental starts (`p` points → OLS fit)
2. C-step: refit on the `h` smallest-residual points, iterate
3. Best 10 subsets → C-step to convergence
4. Return best fit

## Files

- `python/least_trimmed_squares.py` — n=100, p=3
  regression with **30 wild outliers injected**. OLS
  breaks (‖β − β_true‖ = 1.80); LTS at α=0.75 recovers
  `‖β − β_true‖ = 0.21` — nearly 10× improvement under
  30 % contamination.
- `r/least_trimmed_squares.R` — `robustbase::ltsReg` (the
  Rousseeuw reference), `MASS::lqs(method='lts')` (R);
  `sklearn.linear_model.HuberRegressor` / `RANSACRegressor`
  (similar robust flavours), from-scratch (Python).

## When to use

- **Regression with unknown / heavy contamination** — up to
  50 % breakdown.
- **Outlier detection** — LTS residuals expose the
  contamination points.
- **Preliminary step before OLS** on a cleaned subset.
- **Multivariate location / scale via MCD** — the multivariate
  cousin (see mcd-robust-covariance).

## When NOT to use

- **Clean data** — OLS is efficient (Cramér-Rao at Gaussian);
  LTS is 8 % efficient at Gaussian at α=0.5.
- **Very large n and p** — combinatorial search cost;
  FAST-LTS mitigates but is still O(n log n · p³) per subset.
- **Heteroscedastic clean errors** — use weighted OLS or GLS,
  not LTS.

## Assumptions & caveats

- **Trimming fraction α** — 0.5 for max breakdown, 0.75 for
  higher efficiency at clean Gaussian.
- **`h` choice** — `h = ⌊α · n⌋`; often re-weighted OLS on the
  h-subset gives further efficiency (Rousseeuw-Van Driessen).
- **Elemental starts** — need enough to have at least one
  outlier-free elemental subset with high probability
  (Rousseeuw's formula).
- **Consistency correction** — MAD-style scale must be
  Fisher-consistent at Gaussian; `robustbase::ltsReg`
  applies it automatically.

## Related in this repo

- `mcd-robust-covariance` — multivariate cousin.
- `mm-estimators-robust`, `robust-regression`,
  `theil-sen-slope`, `winsorization`, `hampel-identifier`
  — related robust methods.
- `quantile-regression`, `censored-quantile-regression` —
  quantile-based robust alternatives.
- `random-forest`, `gradient-boosting` — implicit robustness
  via tree splits.

## Run

```
python techniques/least-trimmed-squares/python/least_trimmed_squares.py
Rscript techniques/least-trimmed-squares/r/least_trimmed_squares.R
```

**Refs:** Rousseeuw, P.J. "Least median of squares regression." *J. Amer. Statist. Assoc.*, 79: 871-880, 1984; Rousseeuw, P.J. and Van Driessen, K. "Computing LTS regression for large data sets." *Data Mining and Knowl. Discov.*, 12: 29-45, 2006.

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

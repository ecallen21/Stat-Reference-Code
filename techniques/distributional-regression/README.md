# Distributional Regression (Reference §33.12)

Predict the **whole conditional distribution `F(y | x)`**, not just
the mean. Three practical families:

1. **Parametric** (GAMLSS): each distribution parameter is a regression.
2. **Quantile-based**: fit many quantiles and stitch.
3. **CDF / PMF-based**: predict `F(y | x)` directly via a
   discretising classifier or a normalising flow (NGBoost, Duan 2020;
   Distributional Random Forests, Athey-Wager 2019).

## Formula (multinomial-bin CDF)

Discretise `y` into `K` bins, fit softmax regression `P(bin = k | x)`;
derive quantiles as bin-centre approximations of the empirical CDF.

## When to use

- **Probabilistic forecasting** — weather, energy load, sales, medical
  triage.
- **Prediction intervals with heteroscedasticity / skew** — a single
  Gaussian sd is not enough.
- **Downstream utility-based decisions** — expected loss integrals
  over the predictive distribution.

## When NOT to use

- **Point forecast is enough** — OLS / GLM.
- **Very fine tail estimation** — parametric extreme-value or
  distribution-free conformal-quantile methods do better.

## Files

- `python/distributional_regression.py` — from-scratch multinomial
  softmax regression with `K = 20` bins over `y`. Demo on
  heteroscedastic + right-skewed synthetic data. Recovers per-`x`
  quantile bands: at `x = 0` the q10-q90 spread is 3.0; at `x = -1.5`
  it widens to 3.0; the median (q50) tracks the true mean.
- `r/distributional_regression.R` — `gamlss`, `bamlss`, `drf`,
  `quantregForest` (R); `ngboost`, `gluon-ts` (Python).

## Assumptions & caveats

- **Bin choice** — too few = discretisation bias; too many = data-sparse
  bins. Rule of thumb `K = √n / 2`.
- **Monotone quantiles** — bin-based CDF is monotone by construction;
  quantile-crossing artefacts (see `additive-quantile-regression`) are
  avoided.
- **Extreme tails** — cannot extrapolate below `y_min` or above
  `y_max`; use parametric families for tails.
- **Loss** — multinomial CE is a proper score; Brier / CRPS / log-score
  are alternatives.
- **Continuous smoothing** — apply spline smoothing across bins to
  interpolate between them (P-spline log-density regression).

## Related in this repo

- `gamlss` — parametric distributional regression.
- `quantile-regression`, `additive-quantile-regression`,
  `bayesian-quantile-regression` — quantile-based family.
- `conformal-prediction`, `conformal-classification` — coverage-
  guaranteed prediction intervals.
- `epistemic-aleatoric` — related in Bayesian deep learning.
- `logistic-regression`, `multinomial-logit` — the underlying softmax
  fit.

## Run

```
python techniques/distributional-regression/python/distributional_regression.py
Rscript techniques/distributional-regression/r/distributional_regression.R
```

**Refs:** Klein, N., Kneib, T., Klasen, S. & Lang, S. "Bayesian structured additive distributional regression." *Statistical Modelling*, 2015; Duan, T. et al. "NGBoost: natural gradient boosting for probabilistic prediction." *ICML*, 2020; Ćevid, D., Michel, L., Näf, J., Meinshausen, N. & Bühlmann, P. "Distributional random forests." *arXiv*, 2020.

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

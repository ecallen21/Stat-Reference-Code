# Jackknife+ Prediction Intervals (Reference Ch 29 UQ)

**Distribution-free prediction intervals** by leave-one-out refitting.
Barber, Candès, Ramdas & Tibshirani (2021) proved that jackknife+
delivers `P(y_new ∈ interval) ≥ 1 − 2α` **without** any splitting
overhead, and in practice coverage is very close to `1 − α`.

## Algorithm

For each training index `i = 1..n`:

```
μ̂₋ᵢ  = fit base model on training set with row i deleted
Rᵢ    = |yᵢ − μ̂₋ᵢ(xᵢ)|            (LOO residual)
```

At level `1 − α`, the jackknife+ interval for a new `x*` is

```
[  Q_α    ({ μ̂₋ᵢ(x*) − Rᵢ }ᵢ ),
   Q_{1-α}({ μ̂₋ᵢ(x*) + Rᵢ }ᵢ ) ]
```

**CV+** replaces LOO with K-fold: much cheaper for expensive models.

## Coverage guarantee (Barber 2021, Thm 1)

`P(y_new ∈ Ĉ(x_new)) ≥ 1 − 2α`.

Empirically the gap is usually only a few percent above `1 − α`, so the
`2α` factor is a worst-case bound rather than a practical target.

## When to use

- **Any regression** where you want prediction intervals with a proof
  attached and you can afford `n` (or `K`) refits.
- **Small datasets** — split-conformal wastes precious samples on the
  calibration split; jackknife+ uses them all.
- **Nonparametric / random-forest / gradient-boost regressors** — LOO refits
  are cheap enough for K-fold CV+.

## Files

- `python/jackknife_plus.py` — from-scratch: **Sherman-Morrison LOO
  update** for linear regression (avoids retraining n times); 200 trials
  on synthetic Gaussian regression (n = 40, d = 3). Result:
  **target coverage 0.90 → empirical 0.917** (guarantee ≥ 0.80).
- `r/jackknife_plus.R` — `conformalInference` R package for the same
  procedure; `mapie` / `nonconformist` / `puncc` in Python.

## Assumptions & caveats

- **Refit cost is n × base-model cost** — use CV+ (K = 5–10) for
  expensive models like deep nets or boosted trees.
- **Exchangeability required** — heteroscedastic noise widens the interval
  everywhere (interval is *marginal*, not adaptive). See
  `conformal-prediction` / CQR for locally-adaptive width.
- **Distribution shift** — the guarantee breaks; use weighted conformal
  (`covariate-shift-adaptation`) or robust variants.
- **Numeric stability of LOO refit** — Sherman-Morrison denominator
  `1 − hᵢᵢ` can be small when a leverage point is present. Guard with
  ridge or drop rows with hᵢᵢ ≈ 1.

## Related in this repo

- `jackknife` — LOO SE + bias correction (this technique's parent).
- `conformal-prediction`, `conformal-classification` — split-conformal
  alternatives.
- `bootstrap`, `bca-bootstrap` — resampling-based inference alternatives.
- `cross-validation` — CV+ is jackknife+'s K-fold cousin.

## Run

```
python techniques/jackknife-plus/python/jackknife_plus.py
Rscript techniques/jackknife-plus/r/jackknife_plus.R
```

**Refs:** Barber, R.F., Candès, E.J., Ramdas, A. & Tibshirani, R.J. "Predictive inference with the jackknife+." *Annals of Statistics*, 49(1), 2021; Lei, J. et al. "Distribution-free predictive inference for regression." *JASA*, 2018.

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
